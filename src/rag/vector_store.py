"""
Vector store implementation using ChromaDB for RAG system.
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from ..utils.document_processor import DocumentChunk

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for document embeddings using ChromaDB."""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "iam_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: Name of the sentence transformer model
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Error loading embedding model {embedding_model}: {e}")
            # Fallback to a simpler model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Using fallback embedding model: all-MiniLM-L6-v2")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "IAM documentation chunks for RAG system"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects to add
        """
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # Ensure chunk_id is unique
            unique_id = f"{chunk.chunk_id}_{uuid.uuid4().hex[:8]}"
            
            documents.append(chunk.content)
            metadatas.append({
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "content_length": len(chunk.content)
            })
            ids.append(unique_id)
        
        # Generate embeddings
        embeddings = self._generate_embeddings(documents)
        
        # Add to ChromaDB collection
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings.tolist()
            )
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        try:
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Perform similarity search to find relevant document chunks.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of search results with content, metadata, and similarity scores
        """
        if not query.strip():
            logger.warning("Empty query provided for similarity search")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self._generate_embeddings([query])[0]
            
            # Perform search
            where_clause = filter_metadata if filter_metadata else None
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for doc, metadata, distance in zip(documents, metadatas, distances):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity = 1 - distance
                    
                    if similarity >= min_similarity:
                        search_results.append({
                            "content": doc,
                            "metadata": metadata,
                            "similarity": similarity,
                            "distance": distance
                        })
                
            search_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Similarity search returned {len(search_results)} results for query: '{query[:50]}...'")
            return search_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get some sample documents to analyze
            sample_results = self.collection.peek(limit=10)
            
            # Analyze topics
            topics = {}
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    topic = metadata.get('topic', 'Unknown')
                    topics[topic] = topics.get(topic, 0) + 1
            
            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "embedding_model": self.embedding_model_name,
                "persist_directory": str(self.persist_directory),
                "sample_topics": topics
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def reset_collection(self) -> None:
        """Reset the collection (delete and recreate)."""
        try:
            # Delete existing collection
            try:
                self.client.delete_collection(name=self.collection_name)
            except ValueError:
                pass  # Collection doesn't exist
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "IAM documentation chunks for RAG system"}
            )
            logger.info(f"Reset collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

class RAGRetriever:
    """High-level RAG retriever that combines vector search with ranking."""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
        self.iam_keywords = [
            "autenticación", "autorización", "oauth", "openid", "pkce",
            "sesión", "token", "criptografía", "acceso", "usuario",
            "identidad", "seguridad", "contraseña", "multi-factor",
            "saml", "jwt", "api", "endpoint", "vulnerabilidad"
        ]
    
    def retrieve_context(
        self,
        query: str,
        max_chunks: int = 5,
        min_similarity: float = 0.3,
        boost_iam_terms: bool = True
    ) -> Tuple[List[str], List[Dict]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            max_chunks: Maximum number of chunks to retrieve
            min_similarity: Minimum similarity threshold
            boost_iam_terms: Whether to boost IAM-related terms
            
        Returns:
            Tuple of (context_texts, metadata_list)
        """
        # Enhance query with IAM context if needed
        enhanced_query = self._enhance_query(query) if boost_iam_terms else query
        
        # Perform similarity search
        results = self.vector_store.similarity_search(
            query=enhanced_query,
            n_results=max_chunks * 2,  # Get more results for reranking
            min_similarity=min_similarity
        )
        
        # Rerank results based on IAM relevance
        reranked_results = self._rerank_by_iam_relevance(results, query)
        
        # Take top results
        top_results = reranked_results[:max_chunks]
        
        # Extract context and metadata
        context_texts = [result["content"] for result in top_results]
        metadata_list = [result["metadata"] for result in top_results]
        
        logger.info(f"Retrieved {len(context_texts)} context chunks for query")
        
        return context_texts, metadata_list
    
    def _enhance_query(self, query: str) -> str:
        """
        Enhance query with IAM-specific context and English translation.
        
        Args:
            query: Original query
            
        Returns:
            Enhanced query
        """
        query_lower = query.lower()
        
        # Translate Spanish terms to English for better embedding matching
        spanish_to_english = {
            "¿qué es": "what is",
            "¿cómo": "how",
            "¿cuál": "which",
            "¿cuáles son": "what are",
            "¿por qué": "why",
            "oauth2": "oauth2",
            "autenticación": "authentication",
            "autorización": "authorization", 
            "sesión": "session",
            "token": "token",
            "contraseña": "password",
            "usuario": "user",
            "identidad": "identity",
            "seguridad": "security",
            "acceso": "access",
            "multi-factor": "multi-factor",
            "saml": "saml",
            "jwt": "jwt",
            "pkce": "pkce",
            "openid": "openid"
        }
        
        # Apply translations
        enhanced_query = query_lower
        for spanish, english in spanish_to_english.items():
            enhanced_query = enhanced_query.replace(spanish, english)
        
        # Remove Spanish punctuation
        enhanced_query = enhanced_query.replace("¿", "").replace("?", "")
        
        # Add IAM context if not present  
        if not any(keyword in enhanced_query for keyword in self.iam_keywords):
            enhanced_query = f"{enhanced_query} IAM security identity management"
        
        return enhanced_query.strip()
    
    def _rerank_by_iam_relevance(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Rerank results based on IAM relevance.
        
        Args:
            results: Search results
            query: Original query
            
        Returns:
            Reranked results
        """
        query_lower = query.lower()
        
        for result in results:
            content_lower = result["content"].lower()
            topic = result["metadata"].get("topic", "").lower()
            
            # Base score is similarity
            score = result["similarity"]
            
            # Boost for IAM keyword matches in content
            iam_matches = sum(1 for keyword in self.iam_keywords 
                            if keyword in content_lower)
            score += iam_matches * 0.1
            
            # Boost for query term matches
            query_terms = query_lower.split()
            term_matches = sum(1 for term in query_terms 
                             if term in content_lower)
            score += (term_matches / len(query_terms)) * 0.2
            
            # Boost for topic relevance
            if any(keyword in topic for keyword in query_terms):
                score += 0.15
            
            result["final_score"] = score
        
        # Sort by final score
        return sorted(results, key=lambda x: x["final_score"], reverse=True)

def main():
    """Main function for testing vector store."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Print collection info
    info = vector_store.get_collection_info()
    print("Collection Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test search if collection has data
    if info.get("total_chunks", 0) > 0:
        print("\n--- Test Similarity Search ---")
        
        test_queries = [
            "¿Qué tipos de autenticación existen?",
            "¿Cómo funciona OAuth2?",
            "¿Qué es PKCE y cuándo se usa?"
        ]
        
        retriever = RAGRetriever(vector_store)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            context_texts, metadata_list = retriever.retrieve_context(query, max_chunks=3)
            
            for i, (context, metadata) in enumerate(zip(context_texts, metadata_list)):
                print(f"  Result {i+1}:")
                print(f"    Topic: {metadata.get('topic')}")
                print(f"    Source: {metadata.get('source')}")
                print(f"    Content: {context[:150]}...")
    else:
        print("No documents in collection. Run document processing first.")

if __name__ == "__main__":
    main()
