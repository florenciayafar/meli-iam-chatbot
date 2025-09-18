#!/usr/bin/env python3
"""
Setup script to initialize the vector database and process IAM documents.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.document_processor import DocumentProcessor
from src.rag.vector_store import VectorStore
from src.bot.llm_interface import OllamaLLMInterface

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from environment or config file."""
    # Try to load from config file if it exists
    config_file = Path("config.env")
    if config_file.exists():
        logger.info("Loading configuration from config.env")
        with open(config_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, _, value = line.strip().partition('=')
                    os.environ[key] = value

    return {
        'documents_dir': Path(os.getenv('DOCUMENTS_DIR', './data/documents')),
        'chroma_dir': os.getenv('CHROMA_PERSIST_DIRECTORY', './data/chroma'),
        'collection_name': os.getenv('CHROMA_COLLECTION_NAME', 'iam_documents'),
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
        'chunk_size': int(os.getenv('CHUNK_SIZE', '1000')),
        'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '200')),
        'min_chunk_size': int(os.getenv('MIN_CHUNK_SIZE', '100')),
        'ollama_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        'llm_model': os.getenv('LLM_MODEL', 'llama3')
    }

async def check_ollama_service(config):
    """Check if Ollama service is available and pull model if needed."""
    logger.info("Checking Ollama service...")
    
    try:
        llm = OllamaLLMInterface(
            base_url=config['ollama_url'],
            model=config['llm_model']
        )
        
        is_healthy = await llm.check_health()
        
        if not is_healthy:
            logger.info(f"Model '{config['llm_model']}' not available. Attempting to pull...")
            success = await llm.pull_model()
            
            if success:
                logger.info("Model pulled successfully")
            else:
                logger.warning("Failed to pull model. Please ensure Ollama is running and try manually:")
                logger.warning(f"ollama pull {config['llm_model']}")
        else:
            logger.info(f"Ollama service is ready with model '{config['llm_model']}'")
        
        await llm.close()
        return is_healthy
        
    except Exception as e:
        logger.error(f"Error checking Ollama service: {e}")
        logger.info("Please ensure Ollama is installed and running:")
        logger.info("  1. Install: https://ollama.ai/")
        logger.info(f"  2. Run: ollama pull {config['llm_model']}")
        logger.info("  3. Start service: ollama serve")
        return False

def process_documents_sync(config):
    """Process documents and create chunks."""
    logger.info("Processing IAM documents...")
    
    documents_dir = config['documents_dir']
    if not documents_dir.exists():
        logger.error(f"Documents directory not found: {documents_dir}")
        logger.info("Please ensure IAM PDF documents are in ./data/documents/")
        return []
    
    # List available documents
    pdf_files = list(documents_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF documents:")
    for pdf in pdf_files:
        logger.info(f"  - {pdf.name}")
    
    if not pdf_files:
        logger.error("No PDF files found in documents directory")
        return []
    
    # Initialize document processor
    processor = DocumentProcessor(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        min_chunk_size=config['min_chunk_size']
    )
    
    try:
        # Process all documents
        chunks = processor.process_documents_directory(documents_dir)
        
        # Save processed data summary
        processed_dir = Path("./data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        processor.save_processed_chunks(chunks, processed_dir / "chunks_summary.json")
        
        logger.info(f"Successfully processed {len(chunks)} chunks from {len(pdf_files)} documents")
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return []

def setup_vector_database(config, chunks):
    """Initialize vector database and add document chunks."""
    logger.info("Setting up vector database...")
    
    try:
        # Create directories
        chroma_dir = Path(config['chroma_dir'])
        chroma_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize vector store
        vector_store = VectorStore(
            persist_directory=config['chroma_dir'],
            collection_name=config['collection_name'],
            embedding_model=config['embedding_model']
        )
        
        # Check if collection already has data
        collection_info = vector_store.get_collection_info()
        existing_chunks = collection_info.get('total_chunks', 0)
        
        if existing_chunks > 0:
            logger.info(f"Vector database already contains {existing_chunks} chunks")
            
            # Ask user if they want to reset
            response = input("Do you want to reset the database and re-add documents? (y/N): ")
            if response.lower().startswith('y'):
                logger.info("Resetting vector database...")
                vector_store.reset_collection()
            else:
                logger.info("Keeping existing data")
                return vector_store
        
        if chunks:
            # Add chunks to vector store
            logger.info("Adding document chunks to vector database...")
            vector_store.add_documents(chunks)
            
            # Verify addition
            final_info = vector_store.get_collection_info()
            logger.info(f"Vector database now contains {final_info.get('total_chunks', 0)} chunks")
            
            # Print summary
            print("\n" + "="*50)
            print("VECTOR DATABASE SETUP COMPLETE")
            print("="*50)
            print(f"Collection: {final_info.get('collection_name')}")
            print(f"Total chunks: {final_info.get('total_chunks')}")
            print(f"Embedding model: {final_info.get('embedding_model')}")
            print(f"Storage location: {final_info.get('persist_directory')}")
            print("="*50)
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Error setting up vector database: {e}")
        return None

async def test_system(config):
    """Test the complete system integration."""
    logger.info("Testing system integration...")
    
    try:
        # Test vector store
        vector_store = VectorStore(
            persist_directory=config['chroma_dir'],
            collection_name=config['collection_name']
        )
        
        # Test search
        test_queries = [
            "¿Qué tipos de autenticación existen?",
            "¿Cómo funciona OAuth2?",
            "¿Qué es PKCE?"
        ]
        
        from src.rag.vector_store import RAGRetriever
        retriever = RAGRetriever(vector_store)
        
        print("\n" + "="*50)
        print("SYSTEM TEST RESULTS")
        print("="*50)
        
        for query in test_queries:
            print(f"\nTest query: {query}")
            context_texts, metadata = retriever.retrieve_context(query, max_chunks=2)
            
            if context_texts:
                print(f" Retrieved {len(context_texts)} relevant chunks")
                for i, meta in enumerate(metadata[:1]):  # Show first result
                    print(f"   Source: {meta.get('source')}")
                    print(f"   Topic: {meta.get('topic')}")
            else:
                print(" No relevant content found")
        
        # Test LLM connection
        llm = OllamaLLMInterface(
            base_url=config['ollama_url'],
            model=config['llm_model']
        )
        
        llm_healthy = await llm.check_health()
        print(f"\nLLM Service: {'Ready' if llm_healthy else ' Not available'}")
        
        await llm.close()
        
        print("\n" + "="*50)
        print("Setup completed successfully!")
        print("You can now start the API server with: python -m src.api.main")
        print("Or run the frontend with: streamlit run src/frontend/app.py")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error testing system: {e}")

async def main():
    """Main setup function."""
    print("🚀 MeLi IAM Chatbot - Database Setup")
    print("="*50)
    
    # Load configuration
    config = load_config()
    
    print("Configuration:")
    print(f"  Documents directory: {config['documents_dir']}")
    print(f"  ChromaDB directory: {config['chroma_dir']}")
    print(f"  Collection name: {config['collection_name']}")
    print(f"  Embedding model: {config['embedding_model']}")
    print(f"  LLM model: {config['llm_model']}")
    print()
    
    # Check Ollama service
    await check_ollama_service(config)
    
    # Process documents
    chunks = process_documents_sync(config)
    
    if not chunks:
        logger.error("No document chunks were created. Please check your documents.")
        return
    
    # Setup vector database
    vector_store = setup_vector_database(config, chunks)
    
    if vector_store is None:
        logger.error("Failed to setup vector database")
        return
    
    # Test system
    await test_system(config)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)
