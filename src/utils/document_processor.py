"""
Document processor for extracting and chunking text from IAM documentation PDFs.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass

import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    content: str
    metadata: Dict
    chunk_id: str
    source: str
    page_number: int = None

class DocumentProcessor:
    """Processes PDF documents and creates chunks for vector storage."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, min_chunk_size: int = 100):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks  
            min_chunk_size: Minimum size for a chunk to be considered valid
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
        )
        
        # Document metadata mapping
        self.document_topics = {
            "AccessControl.pdf": "Control de Acceso",
            "Autenticacion+Autorizacion.pdf": "Autenticación y Autorización", 
            "BuenasPracticasTiempoDeSesion2025.pdf": "Mejores Prácticas de Sesión",
            "Criptografia.pdf": "Criptografía y Seguridad",
            "LifeCycle.pdf": "Ciclo de Vida de Identidades",
            "OAuth2.pdf": "OAuth 2.0",
            "OpenIDC.pdf": "OpenID Connect",
            "PKCE.pdf": "Proof Key for Code Exchange",
            "TiposDeAutenticacion.pdf": "Tipos de Autenticación",
            "UserProvisioning.pdf": "Aprovisionamiento de Usuarios"
        }
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Tuple[str, int]]:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of tuples (text_content, page_number)
        """
        pages_text = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text.strip():  # Only add non-empty pages
                            # Clean extracted text
                            text = self._clean_text(text)
                            pages_text.append((text, page_num))
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num} of {pdf_path.name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path.name}: {e}")
            raise
            
        return pages_text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\n?\d+\n?', '', text)
        
        # Remove special characters but keep Spanish accents
        text = re.sub(r'[^\w\s\.\?\!\;\,\:\-\(\)\"\'áéíóúñüÁÉÍÓÚÑÜ]', '', text)
        
        # Normalize line breaks
        text = text.replace('\n', ' ')
        
        return text.strip()
    
    def create_chunks(self, pages_text: List[Tuple[str, int]], source_file: str) -> List[DocumentChunk]:
        """
        Create chunks from extracted text.
        
        Args:
            pages_text: List of (text, page_number) tuples
            source_file: Name of the source file
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        # Combine all text while keeping track of page boundaries
        full_text = ""
        page_map = {}  # Maps character position to page number
        current_pos = 0
        
        for text, page_num in pages_text:
            page_map[current_pos] = page_num
            full_text += text + "\n\n"
            current_pos = len(full_text)
        
        # Split into chunks
        text_chunks = self.text_splitter.split_text(full_text)
        
        # Create DocumentChunk objects
        topic = self.document_topics.get(source_file, "IAM General")
        
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) >= self.min_chunk_size:
                # Find the page number for this chunk
                chunk_start_pos = full_text.find(chunk_text)
                page_number = self._get_page_for_position(page_map, chunk_start_pos)
                
                chunk = DocumentChunk(
                    content=chunk_text.strip(),
                    metadata={
                        "source": source_file,
                        "topic": topic,
                        "chunk_index": i,
                        "page": page_number,
                        "total_chunks": len(text_chunks),
                        "document_type": "IAM_Documentation"
                    },
                    chunk_id=f"{source_file}_{i}",
                    source=source_file,
                    page_number=page_number
                )
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks
    
    def _get_page_for_position(self, page_map: Dict[int, int], position: int) -> int:
        """Get the page number for a given text position."""
        if not page_map:
            return 1
            
        # Find the largest position that is <= given position
        valid_positions = [pos for pos in page_map.keys() if pos <= position]
        if valid_positions:
            return page_map[max(valid_positions)]
        return 1
    
    def process_documents_directory(self, documents_dir: Path) -> List[DocumentChunk]:
        """
        Process all PDF documents in a directory.
        
        Args:
            documents_dir: Directory containing PDF files
            
        Returns:
            List of all document chunks
        """
        all_chunks = []
        pdf_files = list(documents_dir.glob("*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {documents_dir}")
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}...")
                
                # Extract text from PDF
                pages_text = self.extract_text_from_pdf(pdf_file)
                
                if not pages_text:
                    logger.warning(f"No text extracted from {pdf_file.name}")
                    continue
                
                # Create chunks
                file_chunks = self.create_chunks(pages_text, pdf_file.name)
                all_chunks.extend(file_chunks)
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(pdf_files)} documents, created {len(all_chunks)} total chunks")
        return all_chunks
    
    def save_processed_chunks(self, chunks: List[DocumentChunk], output_file: Path) -> None:
        """
        Save processed chunks to a file for inspection.
        
        Args:
            chunks: List of document chunks
            output_file: Path to save the processed data
        """
        import json
        
        chunks_data = []
        for chunk in chunks:
            chunk_data = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "metadata": chunk.metadata,
                "content_length": len(chunk.content)
            }
            chunks_data.append(chunk_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed chunks summary to {output_file}")

def main():
    """Main function for testing document processing."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    # Process documents
    documents_dir = Path("../../data/documents")
    if documents_dir.exists():
        chunks = processor.process_documents_directory(documents_dir)
        
        # Save summary
        output_dir = Path("../../data/processed")
        output_dir.mkdir(exist_ok=True)
        processor.save_processed_chunks(chunks, output_dir / "chunks_summary.json")
        
        print(f"Processed {len(chunks)} chunks from IAM documents")
        
        # Print sample chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.source}")
            print(f"Topic: {chunk.metadata['topic']}")
            print(f"Content preview: {chunk.content[:200]}...")
    else:
        print(f"Documents directory not found: {documents_dir}")

if __name__ == "__main__":
    main()
