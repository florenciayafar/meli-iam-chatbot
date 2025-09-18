"""
Pydantic models for API request/response validation.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import re

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message to send to the chatbot"
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique session identifier for conversation continuity"
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include document sources in response"
    )
    max_context_chunks: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of document chunks to retrieve for context"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="LLM temperature for response generation"
    )
    
    @validator("message")
    def validate_message(cls, v):
        """Validate and clean message content."""
        # Remove excessive whitespace
        v = re.sub(r'\s+', ' ', v.strip())
        
        # Check for empty message after cleaning
        if not v:
            raise ValueError("Message cannot be empty")
        
        return v
    
    @validator("session_id")
    def validate_session_id(cls, v):
        """Validate session ID format."""
        # Allow alphanumeric, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Session ID must contain only alphanumeric characters, hyphens, and underscores")
        
        return v

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(
        ...,
        description="Chatbot response to user message"
    )
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Response metadata including timing, sources, etc."
    )

class MessageInfo(BaseModel):
    """Model for individual message in conversation history."""
    content: str
    role: str = Field(..., pattern=r"^(user|assistant)$")
    timestamp: float
    metadata: Dict[str, Any] = {}
    tokens_used: int = 0
    
    @property
    def formatted_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.fromtimestamp(self.timestamp).isoformat()

class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    session_id: str
    messages: List[MessageInfo]
    total_messages: int
    session_stats: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., pattern=r"^(healthy|degraded|unhealthy)$")
    timestamp: str
    version: str = "1.0.0"
    components: Dict[str, Dict[str, Any]]
    uptime_seconds: float

class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    api_status: str
    llm_status: Dict[str, Any]
    vector_db_status: Dict[str, Any]
    memory_status: Dict[str, Any]
    system_info: Dict[str, Any]
    statistics: Dict[str, Any]

class DocumentSource(BaseModel):
    """Model for document source information."""
    source: str
    topic: str
    page: Optional[int] = None
    relevance_score: Optional[float] = None

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: Optional[str] = None

# Request models for additional endpoints
class SessionDeleteRequest(BaseModel):
    """Request to delete a conversation session."""
    session_id: str = Field(..., min_length=1, max_length=100)
    confirm: bool = Field(
        default=False,
        description="Confirmation flag to prevent accidental deletion"
    )
    
    @validator("confirm")
    def validate_confirm(cls, v):
        if not v:
            raise ValueError("Confirmation required to delete session")
        return v

class BulkChatRequest(BaseModel):
    """Request model for bulk chat processing (for testing)."""
    messages: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of messages to process"
    )
    session_id: str
    
    @validator("messages")
    def validate_messages(cls, v):
        for msg in v:
            if not msg.strip():
                raise ValueError("All messages must be non-empty")
        return v

class SystemConfigRequest(BaseModel):
    """Request to update system configuration."""
    llm_temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, ge=100, le=4000)
    max_context_chunks: Optional[int] = Field(None, ge=1, le=20)
    memory_strategy: Optional[str] = Field(None, pattern=r"^(buffer|summary|vector)$")
    
class AnalyticsResponse(BaseModel):
    """Response model for analytics data."""
    session_count: int
    total_messages: int
    avg_response_time_ms: float
    top_queries: List[Dict[str, Any]]
    error_rate: float
    system_performance: Dict[str, Any]
    time_period: str

# Validation utilities
class ValidationUtils:
    """Utility functions for data validation."""
    
    @staticmethod
    def is_valid_session_id(session_id: str) -> bool:
        """Check if session ID is valid."""
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', session_id))
    
    @staticmethod
    def sanitize_message(message: str) -> str:
        """Sanitize user message."""
        # Remove control characters but keep Spanish characters
        message = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', message)
        
        # Normalize whitespace
        message = re.sub(r'\s+', ' ', message.strip())
        
        return message
    
    @staticmethod
    def is_iam_related_query(query: str) -> bool:
        """Check if query is related to IAM topics."""
        iam_keywords = [
            "autenticación", "autorización", "oauth", "openid", "pkce",
            "sesión", "token", "jwt", "saml", "criptografía", "acceso",
            "usuario", "contraseña", "multi-factor", "seguridad", "login",
            "identidad", "permiso", "rol", "política", "api"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in iam_keywords)

# Custom validators for FastAPI
def validate_session_id_param(session_id: str) -> str:
    """Validate session ID path parameter."""
    if not ValidationUtils.is_valid_session_id(session_id):
        raise ValueError("Invalid session ID format")
    return session_id

def validate_message_content(message: str) -> str:
    """Validate message content."""
    sanitized = ValidationUtils.sanitize_message(message)
    if not sanitized:
        raise ValueError("Message cannot be empty after sanitization")
    if len(sanitized) > 2000:
        raise ValueError("Message too long")
    return sanitized

# Response builders
class ResponseBuilder:
    """Utility class for building standardized responses."""
    
    @staticmethod
    def success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
        """Build success response."""
        return {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def error_response(
        error: str,
        detail: str = None,
        error_code: str = None,
        status_code: int = 400
    ) -> ErrorResponse:
        """Build error response."""
        return ErrorResponse(
            error=error,
            detail=detail,
            error_code=error_code
        )
    
    @staticmethod
    def health_response(
        status: str,
        components: Dict[str, Dict[str, Any]],
        uptime: float
    ) -> HealthResponse:
        """Build health check response."""
        return HealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            components=components,
            uptime_seconds=uptime
        )

# Configuration models
class APIConfig(BaseModel):
    """API configuration model."""
    title: str = "MeLi IAM Chatbot API"
    version: str = "1.0.0"
    description: str = "API for IAM chatbot with RAG and conversation memory"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10
    
    # CORS
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["GET", "POST", "DELETE"]
    
    # Security
    api_key_header: str = "X-API-Key"
    require_api_key: bool = False
    
    # Logging
    log_level: str = "INFO"
    access_log: bool = True
    
    class Config:
        env_prefix = "API_"

def get_example_responses():
    """Get example responses for OpenAPI documentation."""
    return {
        "chat_success": {
            "response": "OAuth2 es un protocolo de autorización que permite a las aplicaciones obtener acceso limitado a cuentas de usuario...",
            "session_id": "user-123",
            "metadata": {
                "response_time_ms": 1250,
                "llm_response_time_ms": 1100,
                "model_used": "llama3",
                "tokens_used": 245,
                "confidence_score": 0.85,
                "chunks_retrieved": 3,
                "sources_used": ["OAuth2.pdf", "Autenticacion+Autorizacion.pdf"],
                "timestamp": "2024-01-15T10:30:45.123Z"
            }
        },
        "health_success": {
            "status": "healthy",
            "timestamp": "2024-01-15T10:30:45.123Z",
            "version": "1.0.0",
            "components": {
                "llm": {"status": "healthy", "model": "llama3", "response_time_ms": 150},
                "vector_db": {"status": "healthy", "collection_size": 1250},
                "memory": {"status": "healthy", "active_sessions": 5}
            },
            "uptime_seconds": 86400.5
        },
        "error_example": {
            "error": "Validation error",
            "detail": "Message cannot be empty",
            "error_code": "VALIDATION_ERROR",
            "timestamp": "2024-01-15T10:30:45.123Z"
        }
    }
