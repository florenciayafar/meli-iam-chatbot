"""
FastAPI main application for MeLi IAM Chatbot.
"""

import os
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn

# Import our modules
from .models import (
    ChatRequest, ChatResponse, ConversationHistoryResponse,
    HealthResponse, SystemStatusResponse, ErrorResponse,
    ResponseBuilder, APIConfig, get_example_responses
)
from ..bot.llm_interface import OllamaLLMInterface, IAMChatbot
from ..rag.vector_store import VectorStore, RAGRetriever
from ..memory.conversation_memory import ConversationMemory, create_memory_manager
from ..utils.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for app components
chatbot: IAMChatbot = None
app_start_time: float = 0
request_count: int = 0
error_count: int = 0

async def initialize_system():
    """Initialize all system components."""
    global chatbot, app_start_time
    
    try:
        app_start_time = time.time()
        logger.info("Initializing MeLi IAM Chatbot system...")
        
        # Load configuration
        config = APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        )
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "iam_documents"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        # Check if documents need to be processed
        collection_info = vector_store.get_collection_info()
        if collection_info.get("total_chunks", 0) == 0:
            logger.info("No documents found in vector store. Processing documents...")
            await process_documents(vector_store)
        
        # Initialize RAG retriever
        rag_retriever = RAGRetriever(vector_store)
        
        # Initialize memory manager
        memory_strategy = os.getenv("MEMORY_STRATEGY", "buffer")
        memory_manager = create_memory_manager(
            strategy_name=memory_strategy,
            max_tokens=int(os.getenv("MAX_MEMORY_TOKENS", "4000")),
            persist_file=Path("./data/memory/conversations.json"),
            max_session_age_hours=24
        )
        
        # Initialize LLM interface
        llm_interface = OllamaLLMInterface(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("LLM_MODEL", "llama3"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("MAX_TOKENS", "200"))
        )
        
        # Check LLM availability
        if not await llm_interface.check_health():
            logger.warning("LLM not available. Trying to pull model...")
            await llm_interface.pull_model()
        
        # Initialize chatbot
        chatbot = IAMChatbot(
            llm_interface=llm_interface,
            rag_retriever=rag_retriever,
            memory_manager=memory_manager,
            max_context_chunks=int(os.getenv("MAX_CONTEXT_CHUNKS", "3")),
            min_similarity_threshold=float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.2"))
        )
        
        logger.info("System initialization completed successfully!")
        return config
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

async def process_documents(vector_store: VectorStore):
    """Process IAM documents and add to vector store."""
    try:
        documents_dir = Path("./data/documents")
        if not documents_dir.exists():
            logger.warning(f"Documents directory not found: {documents_dir}")
            return
        
        processor = DocumentProcessor(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "100"))
        )
        
        # Process all documents
        chunks = processor.process_documents_directory(documents_dir)
        
        if chunks:
            # Add to vector store
            vector_store.add_documents(chunks)
            logger.info(f"Added {len(chunks)} chunks to vector store")
        else:
            logger.warning("No document chunks were created")
            
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise

async def shutdown_system():
    """Clean up system resources."""
    global chatbot
    
    try:
        logger.info("Shutting down system...")
        
        if chatbot:
            await chatbot.close()
        
        logger.info("System shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    config = await initialize_system()
    app.state.config = config
    yield
    # Shutdown
    await shutdown_system()

# Create FastAPI app
app = FastAPI(
    title=os.getenv("API_TITLE", "MeLi IAM Chatbot API"),
    description="API para chatbot de IAM con RAG y memoria conversacional",
    version=os.getenv("API_VERSION", "1.0.0"),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Request tracking middleware
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics."""
    global request_count, error_count
    
    start_time = time.time()
    request_count += 1
    
    try:
        response = await call_next(request)
        
        # Log request
        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} - "
            f"{response.status_code} - {process_time:.3f}s"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        error_count += 1
        process_time = time.time() - start_time
        
        logger.error(
            f"{request.method} {request.url.path} - ERROR: {e} - {process_time:.3f}s"
        )
        
        return JSONResponse(
            status_code=500,
            content=ResponseBuilder.error_response(
                error="Internal server error",
                detail=str(e),
                error_code="INTERNAL_ERROR"
            ).dict()
        )

# Dependency to get chatbot instance
async def get_chatbot() -> IAMChatbot:
    """Get chatbot instance."""
    if chatbot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized"
        )
    return chatbot

# API Routes

@app.get(
    "/health", 
    summary="Health check endpoint",
    description="Simple health check to verify API is running"
)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "meli-iam-chatbot", 
        "timestamp": time.time(),
        "uptime": time.time() - app_start_time if app_start_time else 0
    }

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    summary="Send message to IAM chatbot",
    description="Send a message to the IAM chatbot and get a response based on documentation",
    responses={
        200: {"description": "Successful response", "model": ChatResponse},
        400: {"description": "Invalid request", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    }
)
async def chat(
    request: ChatRequest,
    bot: IAMChatbot = Depends(get_chatbot)
) -> ChatResponse:
    """Process chat message and return response."""
    try:
        # Validate IAM relevance (optional warning)
        from .models import ValidationUtils
        if not ValidationUtils.is_iam_related_query(request.message):
            logger.warning(f"Non-IAM query received: {request.message[:50]}...")
        
        # Process chat request
        response = await bot.chat(
            user_query=request.message,
            session_id=request.session_id,
            include_sources=request.include_sources,
            max_chunks=request.max_context_chunks,
            temperature=request.temperature
        )
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat: {str(e)}"
        )

@app.get(
    "/api/chat/history/{session_id}",
    response_model=ConversationHistoryResponse,
    summary="Get conversation history",
    description="Retrieve conversation history for a specific session"
)
async def get_conversation_history(
    session_id: str,
    max_messages: int = 50,
    bot: IAMChatbot = Depends(get_chatbot)
) -> ConversationHistoryResponse:
    """Get conversation history for a session."""
    try:
        from .models import validate_session_id_param
        session_id = validate_session_id_param(session_id)
        
        # Get session info
        session_info = await bot.get_session_info(session_id)
        
        messages = []
        if session_info.get("recent_messages"):
            from .models import MessageInfo
            messages = [
                MessageInfo(**msg) 
                for msg in session_info["recent_messages"][-max_messages:]
            ]
        
        return ConversationHistoryResponse(
            session_id=session_id,
            messages=messages,
            total_messages=len(messages),
            session_stats=session_info.get("session_stats")
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving history: {str(e)}"
        )

@app.delete(
    "/api/chat/session/{session_id}",
    summary="Delete conversation session",
    description="Delete a conversation session and all its messages"
)
async def delete_session(
    session_id: str,
    bot: IAMChatbot = Depends(get_chatbot)
) -> Dict[str, Any]:
    """Delete a conversation session."""
    try:
        from .models import validate_session_id_param
        session_id = validate_session_id_param(session_id)
        
        # Delete session from memory
        deleted = bot.memory.delete_session(session_id)
        
        if deleted:
            return ResponseBuilder.success_response(
                data={"session_id": session_id},
                message="Session deleted successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session deletion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting session: {str(e)}"
        )

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of all system components"
)
async def health_check() -> HealthResponse:
    """Comprehensive health check."""
    try:
        components = {}
        overall_status = "healthy"
        
        # Check LLM
        if chatbot and chatbot.llm:
            llm_healthy = await chatbot.llm.check_health()
            components["llm"] = {
                "status": "healthy" if llm_healthy else "unhealthy",
                "model": chatbot.llm.model,
                "base_url": chatbot.llm.base_url
            }
            if not llm_healthy:
                overall_status = "degraded"
        else:
            components["llm"] = {"status": "unhealthy", "error": "Not initialized"}
            overall_status = "unhealthy"
        
        # Check Vector DB
        if chatbot and chatbot.rag and chatbot.rag.vector_store:
            try:
                collection_info = chatbot.rag.vector_store.get_collection_info()
                components["vector_db"] = {
                    "status": "healthy",
                    "collection_name": collection_info.get("collection_name"),
                    "total_chunks": collection_info.get("total_chunks", 0)
                }
            except Exception as e:
                components["vector_db"] = {"status": "unhealthy", "error": str(e)}
                overall_status = "degraded"
        else:
            components["vector_db"] = {"status": "unhealthy", "error": "Not initialized"}
            overall_status = "unhealthy"
        
        # Check Memory System
        if chatbot and chatbot.memory:
            components["memory"] = {
                "status": "healthy",
                "active_sessions": len(chatbot.memory.sessions)
            }
        else:
            components["memory"] = {"status": "unhealthy", "error": "Not initialized"}
            overall_status = "unhealthy"
        
        # System metrics
        uptime = time.time() - app_start_time if app_start_time else 0
        
        return ResponseBuilder.health_response(
            status=overall_status,
            components=components,
            uptime=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return ResponseBuilder.health_response(
            status="unhealthy",
            components={"error": str(e)},
            uptime=0
        )

@app.get(
    "/api/status",
    response_model=SystemStatusResponse,
    summary="System status",
    description="Get detailed system status and metrics"
)
async def system_status() -> SystemStatusResponse:
    """Get detailed system status."""
    try:
        # LLM status
        llm_status = {"available": False, "model": None}
        if chatbot and chatbot.llm:
            llm_status = {
                "available": await chatbot.llm.check_health(),
                "model": chatbot.llm.model,
                "base_url": chatbot.llm.base_url,
                "temperature": chatbot.llm.temperature
            }
        
        # Vector DB status
        vector_db_status = {"available": False}
        if chatbot and chatbot.rag and chatbot.rag.vector_store:
            try:
                info = chatbot.rag.vector_store.get_collection_info()
                vector_db_status = {
                    "available": True,
                    "collection_info": info
                }
            except Exception as e:
                vector_db_status = {"available": False, "error": str(e)}
        
        # Memory status
        memory_status = {"available": False}
        if chatbot and chatbot.memory:
            memory_status = {
                "available": True,
                "active_sessions": len(chatbot.memory.sessions),
                "strategy": chatbot.memory.strategy.__class__.__name__
            }
        
        # System info
        system_info = {
            "uptime_seconds": time.time() - app_start_time if app_start_time else 0,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "platform": os.name
        }
        
        # Statistics
        statistics = {
            "total_requests": request_count,
            "total_errors": error_count,
            "error_rate": (error_count / max(request_count, 1)) * 100
        }
        
        return SystemStatusResponse(
            api_status="operational",
            llm_status=llm_status,
            vector_db_status=vector_db_status,
            memory_status=memory_status,
            system_info=system_info,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving status: {str(e)}"
        )

@app.get(
    "/api/docs/examples",
    summary="API Examples",
    description="Get example requests and responses for API documentation"
)
async def api_examples() -> Dict[str, Any]:
    """Get API examples for documentation."""
    return get_example_responses()

# Root endpoint
@app.get(
    "/",
    summary="API Root",
    description="Welcome message and basic API information"
)
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "message": "MeLi IAM Chatbot API",
        "version": os.getenv("API_VERSION", "1.0.0"),
        "description": "API para chatbot de IAM con RAG y memoria conversacional",
        "docs_url": "/docs",
        "health_url": "/api/health",
        "status_url": "/api/status",
        "uptime_seconds": time.time() - app_start_time if app_start_time else 0
    }

# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="MeLi IAM Chatbot API",
        version="1.0.0",
        description="API para chatbot de IAM con RAG y memoria conversacional para Mercado Libre",
        routes=app.routes,
    )
    
    # Add custom info
    openapi_schema["info"]["contact"] = {
        "name": "MeLi IAM Team",
        "email": "iam@mercadolibre.com"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    # Add server info
    openapi_schema["servers"] = [
        {
            "url": f"http://localhost:{os.getenv('API_PORT', '8000')}",
            "description": "Development server"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ResponseBuilder.error_response(
            error=exc.detail,
            error_code="HTTP_ERROR"
        ).dict()
    )

@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content=ResponseBuilder.error_response(
            error="Validation error",
            detail=str(exc),
            error_code="VALIDATION_ERROR"
        ).dict()
    )

# Main function to run the app
def main():
    """Run the FastAPI application."""
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True
    )

if __name__ == "__main__":
    main()
