"""
LLM interface for the IAM chatbot using Ollama with Llama 3.
"""

import os
import json
import time
from typing import List, Dict, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

import httpx
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Response from LLM with metadata."""
    content: str
    model_used: str
    tokens_used: int
    response_time_ms: int
    confidence_score: float = 0.0
    sources_used: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.sources_used is None:
            self.sources_used = []
        if self.metadata is None:
            self.metadata = {}

class OllamaLLMInterface:
    """Interface to interact with Ollama LLM service."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 120
    ):
        """
        Initialize Ollama LLM interface.
        
        Args:
            base_url: Ollama service base URL
            model: Model name to use
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # HTTP client for API calls
        self.client = httpx.AsyncClient(timeout=timeout)
        
        # Model availability cache
        self.model_available = None
        self.last_health_check = 0
        
        # System prompts for different scenarios
        self.system_prompts = {
            "iam_expert": """Eres un asistente especializado en IAM (Identity and Access Management) de Mercado Libre con acceso a documentación técnica completa.

DOCUMENTACIÓN DISPONIBLE:
- Documentación completa de IAM (3,710 líneas)
- Control de acceso (ACL, RBAC, ABAC, PBAC)
- Autenticación y autorización avanzada
- Gestión de sesiones y mejores prácticas 2025
- Criptografía aplicada a IAM
- OAuth 2.0, OpenID Connect, PKCE
- Tipos de autenticación y factores múltiples
- Ciclo de vida completo de identidades
- Aprovisionamiento de usuarios y roles

INSTRUCCIONES PARA RESPUESTAS MEJORADAS:
1. Usa CONTEXTO ESPECÍFICO de los documentos técnicos
2. Proporciona EJEMPLOS PRÁCTICOS cuando sea posible
3. Explica CONCEPTOS TÉCNICOS con detalle apropiado
4. Menciona MEJORES PRÁCTICAS específicas de la documentación
5. Si hay múltiples enfoques, compáralos basándote en la documentación
6. Incluye REFERENCIAS a secciones específicas cuando sea relevante
7. Para temas complejos, estructura la respuesta en pasos claros

ESPECIALIDADES TÉCNICAS:
- Métodos de autenticación (Knowledge/Possession/Being)
- Protocolos modernos (OAuth 2.0, OIDC, SAML, PKCE)  
- Arquitecturas de control de acceso (ACL, RBAC, ABAC, ReBAC)
- Criptografía IAM (hashing, encryption, digital signatures)
- Gestión avanzada de sesiones y tokens
- Aprovisionamiento automatizado y ciclo de vida
- IA aplicada a control de acceso (AIAC)
- Compliance y auditoría de accesos

FORMATO DE RESPUESTA:
1. Respuesta directa basada en documentación
2. Contexto técnico relevante
3. Ejemplos prácticos si aplica
4. Mejores prácticas relacionadas
5. Referencias a documentación específica

Solo respondo sobre IAM basándome en la documentación proporcionada.""",
            
            "context_aware": """Eres un experto en IAM de Mercado Libre. Usa el CONTEXTO de la conversación y los DOCUMENTOS para dar respuestas precisas y relevantes sobre Identity and Access Management.

REGLAS ESTRICTAS:
- Solo responde sobre temas de IAM de los documentos
- Usa el contexto conversacional para dar respuestas coherentes
- Si no tienes la información, admítelo claramente
- Mantén consistencia con respuestas anteriores"""
        }
    
    async def check_health(self) -> bool:
        """
        Check if Ollama service is available.
        
        Returns:
            True if service is available
        """
        # Cache health check for 30 seconds
        current_time = time.time()
        if self.model_available is not None and (current_time - self.last_health_check) < 30:
            return self.model_available
        
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                self.model_available = self.model in models
                
                if not self.model_available:
                    logger.warning(f"Model '{self.model}' not found. Available models: {models}")
                else:
                    logger.info(f"Model '{self.model}' is available")
            else:
                self.model_available = False
                logger.error(f"Ollama health check failed: {response.status_code}")
                
        except Exception as e:
            self.model_available = False
            logger.error(f"Error checking Ollama health: {e}")
        
        self.last_health_check = current_time
        return self.model_available or False
    
    async def generate_response(
        self,
        user_query: str,
        context: str = "",
        conversation_history: str = "",
        use_context_aware: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response for user query with context.
        
        Args:
            user_query: User's question
            context: RAG context from documents
            conversation_history: Previous conversation context
            use_context_aware: Whether to use context-aware prompting
            **kwargs: Additional parameters for LLM
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        # Check if service is available
        if not await self.check_health():
            return LLMResponse(
                content="Error: El servicio LLM no está disponible. Por favor, verifica que Ollama esté ejecutándose.",
                model_used=self.model,
                tokens_used=0,
                response_time_ms=int((time.time() - start_time) * 1000),
                confidence_score=0.0
            )
        
        # Build prompt
        prompt = self._build_prompt(user_query, context, conversation_history, use_context_aware)
        
        # Extract sources from context
        sources = self._extract_sources(context)
        
        try:
            # Call Ollama API
            response_data = await self._call_ollama_api(prompt, **kwargs)
            
            # Extract response content
            content = response_data.get("response", "").strip()
            
            # Validate response
            if not content:
                content = "Lo siento, no pude generar una respuesta. Por favor, intenta reformular tu pregunta."
            
            # Calculate confidence based on context relevance
            confidence = self._calculate_confidence(user_query, context, content)
            
            # Estimate tokens (rough approximation)
            tokens_used = len(prompt.split()) + len(content.split())
            
            response_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"Generated response in {response_time}ms, ~{tokens_used} tokens")
            
            return LLMResponse(
                content=content,
                model_used=self.model,
                tokens_used=tokens_used,
                response_time_ms=response_time,
                confidence_score=confidence,
                sources_used=sources,
                metadata={
                    "query_length": len(user_query),
                    "context_length": len(context),
                    "has_conversation_history": bool(conversation_history),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            
            return LLMResponse(
                content=f"Error al generar respuesta: {str(e)}",
                model_used=self.model,
                tokens_used=0,
                response_time_ms=int((time.time() - start_time) * 1000),
                confidence_score=0.0
            )
    
    def _build_prompt(
        self,
        user_query: str,
        context: str,
        conversation_history: str,
        use_context_aware: bool
    ) -> str:
        """Build the complete prompt for the LLM."""
        
        # Choose system prompt
        system_prompt = (
            self.system_prompts["context_aware"] 
            if use_context_aware and conversation_history 
            else self.system_prompts["iam_expert"]
        )
        
        prompt_parts = [system_prompt]
        
        # Add document context if available
        if context.strip():
            prompt_parts.append("\nCONTEXTO DE DOCUMENTOS IAM:")
            prompt_parts.append(context)
        
        # Add conversation history if available
        if conversation_history.strip():
            prompt_parts.append("\nHISTORIAL DE CONVERSACIÓN:")
            prompt_parts.append(conversation_history)
        
        # Add current user query
        prompt_parts.append("\nPREGUNTA DEL USUARIO:")
        prompt_parts.append(user_query)
        
        prompt_parts.append("\nRESPUESTA:")
        
        return "\n".join(prompt_parts)
    
    async def _call_ollama_api(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Call Ollama generate API."""
        
        # Prepare request data
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # Get complete response
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
                "top_k": kwargs.get("top_k", 40),
                "top_p": kwargs.get("top_p", 0.9),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
            }
        }
        
        # Make API call
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=request_data
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def _extract_sources(self, context: str) -> List[str]:
        """Extract source information from context."""
        sources = []
        
        # Look for source patterns in context
        lines = context.split('\n')
        for line in lines:
            if 'Fuente:' in line or 'Source:' in line:
                source = line.split(':')[-1].strip()
                if source and source not in sources:
                    sources.append(source)
        
        # Extract document names from context metadata patterns
        import re
        doc_patterns = re.findall(r'(\w+\.pdf)', context)
        for doc in doc_patterns:
            if doc not in sources:
                sources.append(doc)
        
        return sources[:5]  # Limit to 5 sources
    
    def _calculate_confidence(self, query: str, context: str, response: str) -> float:
        """Calculate confidence score for the response."""
        
        # Base confidence
        confidence = 0.5
        
        # Boost if we have context
        if context.strip():
            confidence += 0.2
        
        # Boost for relevant keywords in response
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        keyword_overlap = len(query_words.intersection(response_words))
        if keyword_overlap > 0:
            confidence += min(0.2, keyword_overlap * 0.05)
        
        # Check for IAM-specific terms
        iam_terms = [
            "autenticación", "autorización", "oauth", "openid", "pkce",
            "token", "sesión", "jwt", "saml", "criptografía", "acceso"
        ]
        
        iam_matches = sum(1 for term in iam_terms if term in response.lower())
        if iam_matches > 0:
            confidence += min(0.15, iam_matches * 0.03)
        
        # Penalize if response indicates uncertainty
        uncertainty_phrases = [
            "no tengo información", "no estoy seguro", "no puedo responder",
            "lo siento", "error", "no disponible"
        ]
        
        if any(phrase in response.lower() for phrase in uncertainty_phrases):
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    async def pull_model(self) -> bool:
        """
        Pull/download the model if not available.
        
        Returns:
            True if model is available after pull
        """
        try:
            logger.info(f"Pulling model '{self.model}'...")
            
            response = await self.client.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model, "stream": False}
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully pulled model '{self.model}'")
                self.model_available = True
                return True
            else:
                logger.error(f"Failed to pull model: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()

class IAMChatbot:
    """Main chatbot that combines LLM, RAG, and memory."""
    
    def __init__(
        self,
        llm_interface: OllamaLLMInterface,
        rag_retriever,  # RAGRetriever from vector_store.py
        memory_manager,  # ConversationMemory from conversation_memory.py
        max_context_chunks: int = 5,
        min_similarity_threshold: float = 0.3
    ):
        """
        Initialize IAM chatbot.
        
        Args:
            llm_interface: LLM interface instance
            rag_retriever: RAG retriever instance
            memory_manager: Conversation memory manager
            max_context_chunks: Maximum chunks to retrieve for context
            min_similarity_threshold: Minimum similarity for chunk relevance
        """
        self.llm = llm_interface
        self.rag = rag_retriever
        self.memory = memory_manager
        self.max_context_chunks = max_context_chunks
        self.min_similarity_threshold = min_similarity_threshold
        
        logger.info("IAM Chatbot initialized")
    
    async def chat(
        self,
        user_query: str,
        session_id: str,
        include_sources: bool = True,
        max_chunks: int = None,
        **llm_kwargs
    ) -> Dict[str, Any]:
        """
        Process user query and generate response.
        
        Args:
            user_query: User's question
            session_id: Conversation session ID
            include_sources: Whether to include source information
            **llm_kwargs: Additional LLM parameters
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        try:
            # 1. Retrieve relevant context using RAG
            chunks_to_use = max_chunks if max_chunks is not None else self.max_context_chunks
            context_texts, context_metadata = self.rag.retrieve_context(
                query=user_query,
                max_chunks=chunks_to_use,
                min_similarity=self.min_similarity_threshold
            )
            
            # 2. Get conversation memory
            conversation_history = self.memory.get_context(session_id)
            
            # 3. Build context string
            rag_context = self._build_rag_context(context_texts, context_metadata, include_sources)
            
            # 4. Generate LLM response
            llm_response = await self.llm.generate_response(
                user_query=user_query,
                context=rag_context,
                conversation_history=conversation_history,
                **llm_kwargs
            )
            
            # 5. Add messages to memory
            self.memory.add_message(session_id, user_query, "user")
            self.memory.add_message(
                session_id, 
                llm_response.content, 
                "assistant",
                tokens_used=llm_response.tokens_used,
                confidence=llm_response.confidence_score
            )
            
            # 6. Build final response
            total_time = int((time.time() - start_time) * 1000)
            
            response = {
                "response": llm_response.content,
                "session_id": session_id,
                "metadata": {
                    "response_time_ms": total_time,
                    "llm_response_time_ms": llm_response.response_time_ms,
                    "model_used": llm_response.model_used,
                    "tokens_used": llm_response.tokens_used,
                    "confidence_score": llm_response.confidence_score,
                    "chunks_retrieved": len(context_texts),
                    "sources_used": llm_response.sources_used if include_sources else [],
                    "has_conversation_history": bool(conversation_history),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Add context information if requested
            if include_sources and context_metadata:
                response["metadata"]["document_sources"] = [
                    {
                        "source": meta.get("source"),
                        "topic": meta.get("topic"),
                        "page": meta.get("page")
                    }
                    for meta in context_metadata
                ]
            
            logger.info(f"Chat response generated in {total_time}ms for session {session_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            
            # Add error message to memory
            error_msg = "Lo siento, ocurrió un error al procesar tu consulta. Por favor intenta de nuevo."
            self.memory.add_message(session_id, user_query, "user")
            self.memory.add_message(session_id, error_msg, "assistant")
            
            return {
                "response": error_msg,
                "session_id": session_id,
                "metadata": {
                    "error": str(e),
                    "response_time_ms": int((time.time() - start_time) * 1000),
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def _build_rag_context(
        self,
        context_texts: List[str],
        context_metadata: List[Dict],
        include_sources: bool
    ) -> str:
        """Build formatted context from RAG results."""
        
        if not context_texts:
            return ""
        
        context_parts = []
        
        for i, (text, metadata) in enumerate(zip(context_texts, context_metadata)):
            context_part = f"[Documento {i+1}]\n{text}"
            
            if include_sources:
                source = metadata.get("source", "Desconocido")
                topic = metadata.get("topic", "")
                if topic:
                    context_part += f"\nFuente: {source} - {topic}"
                else:
                    context_part += f"\nFuente: {source}"
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a conversation session."""
        stats = self.memory.get_session_stats(session_id)
        history = self.memory.get_session_history(session_id, max_messages=20)
        
        return {
            "session_stats": stats,
            "recent_messages": history,
            "llm_model": self.llm.model,
            "llm_available": await self.llm.check_health()
        }
    
    async def close(self):
        """Clean up resources."""
        await self.llm.close()
        if self.memory:
            self.memory.shutdown()

def main():
    """Test the LLM interface."""
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test_llm():
        # Initialize LLM
        llm = OllamaLLMInterface(model="llama3")
        
        # Check health
        is_available = await llm.check_health()
        print(f"LLM Available: {is_available}")
        
        if not is_available:
            print("Trying to pull model...")
            pulled = await llm.pull_model()
            print(f"Model pulled: {pulled}")
        
        if is_available or await llm.check_health():
            # Test response
            test_query = "¿Qué tipos de autenticación existen?"
            test_context = "La autenticación puede ser de varios tipos: básica, multi-factor, biométrica..."
            
            response = await llm.generate_response(
                user_query=test_query,
                context=test_context
            )
            
            print(f"\nQuery: {test_query}")
            print(f"Response: {response.content}")
            print(f"Confidence: {response.confidence_score}")
            print(f"Time: {response.response_time_ms}ms")
        
        await llm.close()
    
    asyncio.run(test_llm())

if __name__ == "__main__":
    main()
