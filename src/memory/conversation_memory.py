"""
Custom conversation memory system for the IAM chatbot.
This module implements memory management without using external libraries that have built-in memory.
"""

import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Represents a single message in a conversation."""
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: float
    metadata: Dict[str, Any] = None
    tokens_used: int = 0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp == 0:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """Create message from dictionary."""
        return cls(**data)
    
    def age_minutes(self) -> float:
        """Get message age in minutes."""
        return (time.time() - self.timestamp) / 60

@dataclass
class ConversationSession:
    """Represents a conversation session."""
    session_id: str
    messages: List[Message]
    created_at: float
    last_activity: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at == 0:
            self.created_at = time.time()
        if self.last_activity == 0:
            self.last_activity = time.time()
    
    def add_message(self, message: Message) -> None:
        """Add a message to the session."""
        self.messages.append(message)
        self.last_activity = time.time()
    
    def get_total_tokens(self) -> int:
        """Get total tokens used in this session."""
        return sum(msg.tokens_used for msg in self.messages)
    
    def get_recent_messages(self, max_age_minutes: int = 60) -> List[Message]:
        """Get messages from the last N minutes."""
        cutoff_time = time.time() - (max_age_minutes * 60)
        return [msg for msg in self.messages if msg.timestamp >= cutoff_time]
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationSession':
        """Create session from dictionary."""
        messages = [Message.from_dict(msg_data) for msg_data in data.get('messages', [])]
        return cls(
            session_id=data['session_id'],
            messages=messages,
            created_at=data.get('created_at', 0),
            last_activity=data.get('last_activity', 0),
            metadata=data.get('metadata', {})
        )

class MemoryStrategy(ABC):
    """Abstract base class for memory strategies."""
    
    @abstractmethod
    def get_context(self, session: ConversationSession, max_tokens: int) -> str:
        """Get conversation context within token limit."""
        pass
    
    @abstractmethod
    def should_retain_message(self, message: Message, session: ConversationSession) -> bool:
        """Decide whether to retain a message in memory."""
        pass

class BufferMemoryStrategy(MemoryStrategy):
    """Simple buffer strategy - keeps recent messages within token limit."""
    
    def __init__(self, tokens_per_message: int = 50):
        """
        Initialize buffer memory strategy.
        
        Args:
            tokens_per_message: Estimated tokens per message for calculation
        """
        self.tokens_per_message = tokens_per_message
    
    def get_context(self, session: ConversationSession, max_tokens: int) -> str:
        """Get recent conversation context within token limit."""
        if not session.messages:
            return ""
        
        max_messages = max(1, max_tokens // self.tokens_per_message)
        recent_messages = session.messages[-max_messages:]
        
        context_parts = []
        for msg in recent_messages:
            role_prefix = "Usuario:" if msg.role == "user" else "Asistente:"
            context_parts.append(f"{role_prefix} {msg.content}")
        
        return "\n\n".join(context_parts)
    
    def should_retain_message(self, message: Message, session: ConversationSession) -> bool:
        """Retain all messages in buffer strategy."""
        return True

class SummaryMemoryStrategy(MemoryStrategy):
    """Summary strategy - keeps summary of old messages + recent messages."""
    
    def __init__(self, summary_threshold: int = 10, tokens_per_message: int = 50):
        """
        Initialize summary memory strategy.
        
        Args:
            summary_threshold: Number of messages before creating summary
            tokens_per_message: Estimated tokens per message
        """
        self.summary_threshold = summary_threshold
        self.tokens_per_message = tokens_per_message
        self.conversation_summary = ""
    
    def get_context(self, session: ConversationSession, max_tokens: int) -> str:
        """Get summarized context within token limit."""
        if not session.messages:
            return ""
        
        # If we have many messages, create/update summary
        if len(session.messages) > self.summary_threshold:
            self._update_summary(session)
        
        # Get recent messages
        recent_count = max(3, (max_tokens - len(self.conversation_summary.split())) // self.tokens_per_message)
        recent_messages = session.messages[-recent_count:]
        
        context_parts = []
        
        # Add summary if exists
        if self.conversation_summary:
            context_parts.append(f"Resumen de conversación previa: {self.conversation_summary}")
        
        # Add recent messages
        if recent_messages:
            context_parts.append("Mensajes recientes:")
            for msg in recent_messages:
                role_prefix = "Usuario:" if msg.role == "user" else "Asistente:"
                context_parts.append(f"{role_prefix} {msg.content}")
        
        return "\n\n".join(context_parts)
    
    def _update_summary(self, session: ConversationSession) -> None:
        """Update conversation summary."""
        if len(session.messages) <= self.summary_threshold:
            return
        
        # Get messages to summarize (excluding recent ones)
        messages_to_summarize = session.messages[:-5]  # Keep last 5 recent
        
        if not messages_to_summarize:
            return
        
        # Create simple extractive summary
        topics_mentioned = set()
        key_points = []
        
        for msg in messages_to_summarize:
            content_lower = msg.content.lower()
            
            # Extract IAM topics mentioned
            iam_topics = [
                "autenticación", "autorización", "oauth", "openid", "pkce",
                "sesión", "token", "criptografía", "acceso", "usuario",
                "contraseña", "multi-factor", "saml", "jwt"
            ]
            
            for topic in iam_topics:
                if topic in content_lower:
                    topics_mentioned.add(topic)
            
            # Extract key points (questions and important statements)
            if msg.role == "user" and ("?" in msg.content or "qué" in content_lower):
                key_points.append(f"Pregunta sobre: {msg.content[:100]}...")
        
        # Build summary
        summary_parts = []
        if topics_mentioned:
            summary_parts.append(f"Temas discutidos: {', '.join(topics_mentioned)}")
        if key_points:
            summary_parts.append(f"Preguntas principales: {'; '.join(key_points[:3])}")
        
        self.conversation_summary = ". ".join(summary_parts)
    
    def should_retain_message(self, message: Message, session: ConversationSession) -> bool:
        """Retain recent messages and important ones."""
        # Always retain recent messages
        if message.age_minutes() < 30:
            return True
        
        # Retain questions and answers with IAM keywords
        content_lower = message.content.lower()
        iam_keywords = ["oauth", "openid", "pkce", "jwt", "saml", "autenticación"]
        
        return any(keyword in content_lower for keyword in iam_keywords)

class VectorMemoryStrategy(MemoryStrategy):
    """Vector-based memory strategy - uses embedding similarity for relevance."""
    
    def __init__(self, embedding_model=None, max_relevant_messages: int = 8):
        """
        Initialize vector memory strategy.
        
        Args:
            embedding_model: Sentence transformer model for embeddings
            max_relevant_messages: Maximum number of relevant messages to include
        """
        self.max_relevant_messages = max_relevant_messages
        self.embedding_model = embedding_model
        self.message_embeddings = {}  # Cache for message embeddings
    
    def get_context(self, session: ConversationSession, max_tokens: int) -> str:
        """Get context based on semantic similarity to recent messages."""
        if not session.messages or len(session.messages) < 2:
            return self._get_simple_context(session)
        
        # Get the most recent user message as query
        recent_user_messages = [msg for msg in session.messages if msg.role == "user"]
        if not recent_user_messages:
            return self._get_simple_context(session)
        
        query_message = recent_user_messages[-1]
        
        # Find semantically similar messages
        relevant_messages = self._find_relevant_messages(query_message, session)
        
        # Build context
        context_parts = []
        for msg in relevant_messages:
            role_prefix = "Usuario:" if msg.role == "user" else "Asistente:"
            context_parts.append(f"{role_prefix} {msg.content}")
        
        return "\n\n".join(context_parts)
    
    def _get_simple_context(self, session: ConversationSession) -> str:
        """Fallback to simple context when vector similarity not available."""
        if not session.messages:
            return ""
        
        recent_messages = session.messages[-6:]  # Last 6 messages
        context_parts = []
        for msg in recent_messages:
            role_prefix = "Usuario:" if msg.role == "user" else "Asistente:"
            context_parts.append(f"{role_prefix} {msg.content}")
        
        return "\n\n".join(context_parts)
    
    def _find_relevant_messages(self, query_message: Message, session: ConversationSession) -> List[Message]:
        """Find messages relevant to the query message."""
        if self.embedding_model is None:
            # Fallback to keyword matching
            return self._find_relevant_by_keywords(query_message, session)
        
        # TODO: Implement proper vector similarity
        # For now, use keyword matching as fallback
        return self._find_relevant_by_keywords(query_message, session)
    
    def _find_relevant_by_keywords(self, query_message: Message, session: ConversationSession) -> List[Message]:
        """Find relevant messages using keyword matching."""
        query_words = set(query_message.content.lower().split())
        relevant_messages = []
        
        for msg in session.messages:
            if msg == query_message:
                continue
            
            msg_words = set(msg.content.lower().split())
            overlap = len(query_words.intersection(msg_words))
            
            if overlap >= 2:  # At least 2 common words
                relevant_messages.append((msg, overlap))
        
        # Sort by relevance and take top messages
        relevant_messages.sort(key=lambda x: x[1], reverse=True)
        selected_messages = [msg for msg, _ in relevant_messages[:self.max_relevant_messages]]
        
        # Add the query message and some recent context
        result = session.messages[-3:] + selected_messages
        
        # Remove duplicates while preserving order
        seen = set()
        final_messages = []
        for msg in result:
            if id(msg) not in seen:
                seen.add(id(msg))
                final_messages.append(msg)
        
        return final_messages[-self.max_relevant_messages:]
    
    def should_retain_message(self, message: Message, session: ConversationSession) -> bool:
        """Retain messages based on content importance."""
        content_lower = message.content.lower()
        
        # Always retain recent messages
        if message.age_minutes() < 60:
            return True
        
        # Retain messages with IAM-relevant content
        important_terms = [
            "error", "problema", "configuración", "implementar",
            "oauth", "jwt", "token", "criptografía", "sesión"
        ]
        
        return any(term in content_lower for term in important_terms)

class ConversationMemory:
    """Main conversation memory manager."""
    
    def __init__(
        self,
        strategy: MemoryStrategy = None,
        max_tokens: int = 4000,
        persist_file: Optional[Path] = None,
        cleanup_interval_minutes: int = 60,
        max_session_age_hours: int = 24
    ):
        """
        Initialize conversation memory.
        
        Args:
            strategy: Memory strategy to use
            max_tokens: Maximum tokens for context
            persist_file: File to persist sessions
            cleanup_interval_minutes: How often to cleanup old sessions
            max_session_age_hours: Maximum age for sessions before cleanup
        """
        self.strategy = strategy or BufferMemoryStrategy()
        self.max_tokens = max_tokens
        self.persist_file = persist_file
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.max_session_age_hours = max_session_age_hours
        
        # In-memory storage
        self.sessions: Dict[str, ConversationSession] = {}
        self._lock = threading.Lock()
        
        # Load persisted sessions
        if self.persist_file and self.persist_file.exists():
            self._load_sessions()
        
        # Start cleanup timer
        self._start_cleanup_timer()
    
    def add_message(self, session_id: str, content: str, role: str, **metadata) -> None:
        """
        Add a message to a conversation session.
        
        Args:
            session_id: Session identifier
            content: Message content
            role: Message role ('user' or 'assistant')
            **metadata: Additional message metadata
        """
        with self._lock:
            # Get or create session
            if session_id not in self.sessions:
                self.sessions[session_id] = ConversationSession(
                    session_id=session_id,
                    messages=[],
                    created_at=time.time(),
                    last_activity=time.time()
                )
            
            session = self.sessions[session_id]
            
            # Create message
            message = Message(
                content=content,
                role=role,
                timestamp=time.time(),
                metadata=metadata,
                tokens_used=len(content.split())  # Simple token estimation
            )
            
            # Add message to session
            session.add_message(message)
            
            # Cleanup old messages if needed
            self._cleanup_session_messages(session)
            
            logger.info(f"Added {role} message to session {session_id}")
    
    def get_context(self, session_id: str) -> str:
        """
        Get conversation context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted conversation context
        """
        with self._lock:
            if session_id not in self.sessions:
                return ""
            
            session = self.sessions[session_id]
            return self.strategy.get_context(session, self.max_tokens)
    
    def get_session_history(self, session_id: str, max_messages: int = 50) -> List[Dict]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            max_messages: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        with self._lock:
            if session_id not in self.sessions:
                return []
            
            session = self.sessions[session_id]
            messages = session.messages[-max_messages:] if max_messages else session.messages
            return [msg.to_dict() for msg in messages]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was deleted, False if not found
        """
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Deleted session {session_id}")
                return True
            return False
    
    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """
        Get statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session statistics dictionary
        """
        with self._lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            return {
                "session_id": session_id,
                "message_count": len(session.messages),
                "total_tokens": session.get_total_tokens(),
                "created_at": datetime.fromtimestamp(session.created_at).isoformat(),
                "last_activity": datetime.fromtimestamp(session.last_activity).isoformat(),
                "age_hours": (time.time() - session.created_at) / 3600
            }
    
    def _cleanup_session_messages(self, session: ConversationSession) -> None:
        """Clean up old messages in a session based on strategy."""
        if len(session.messages) <= 20:  # Keep at least 20 messages
            return
        
        # Filter messages based on strategy
        retained_messages = []
        for msg in session.messages:
            if self.strategy.should_retain_message(msg, session):
                retained_messages.append(msg)
        
        # Ensure we keep at least the last 10 messages
        if len(retained_messages) < 10:
            retained_messages = session.messages[-10:]
        
        session.messages = retained_messages
    
    def _cleanup_old_sessions(self) -> None:
        """Remove old inactive sessions."""
        cutoff_time = time.time() - (self.max_session_age_hours * 3600)
        
        with self._lock:
            sessions_to_remove = []
            for session_id, session in self.sessions.items():
                if session.last_activity < cutoff_time:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.sessions[session_id]
                logger.info(f"Cleaned up old session {session_id}")
            
            if sessions_to_remove:
                logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
    
    def _start_cleanup_timer(self) -> None:
        """Start periodic cleanup timer."""
        def cleanup_worker():
            while True:
                time.sleep(self.cleanup_interval_minutes * 60)
                self._cleanup_old_sessions()
                if self.persist_file:
                    self._save_sessions()
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _save_sessions(self) -> None:
        """Save sessions to file."""
        if not self.persist_file:
            return
        
        try:
            with self._lock:
                sessions_data = {}
                for session_id, session in self.sessions.items():
                    sessions_data[session_id] = session.to_dict()
                
                self.persist_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.persist_file, 'w', encoding='utf-8') as f:
                    json.dump(sessions_data, f, indent=2, ensure_ascii=False)
                
                logger.debug(f"Saved {len(sessions_data)} sessions to {self.persist_file}")
                
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def _load_sessions(self) -> None:
        """Load sessions from file."""
        try:
            with open(self.persist_file, 'r', encoding='utf-8') as f:
                sessions_data = json.load(f)
            
            for session_id, session_data in sessions_data.items():
                self.sessions[session_id] = ConversationSession.from_dict(session_data)
            
            logger.info(f"Loaded {len(sessions_data)} sessions from {self.persist_file}")
            
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
    
    def shutdown(self) -> None:
        """Shutdown memory manager and save data."""
        if self.persist_file:
            self._save_sessions()
        logger.info("Conversation memory shutdown complete")

def create_memory_manager(strategy_name: str = "buffer", **kwargs) -> ConversationMemory:
    """
    Factory function to create memory manager with specified strategy.
    
    Args:
        strategy_name: Strategy name ('buffer', 'summary', 'vector')
        **kwargs: Additional arguments for memory manager
        
    Returns:
        ConversationMemory instance
    """
    strategies = {
        "buffer": BufferMemoryStrategy,
        "summary": SummaryMemoryStrategy,
        "vector": VectorMemoryStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    strategy_class = strategies[strategy_name]
    strategy = strategy_class()
    
    return ConversationMemory(strategy=strategy, **kwargs)

def main():
    """Main function for testing memory system."""
    logging.basicConfig(level=logging.INFO)
    
    # Test different strategies
    strategies = ["buffer", "summary"]
    
    for strategy_name in strategies:
        print(f"\n--- Testing {strategy_name} strategy ---")
        
        memory = create_memory_manager(strategy_name, max_tokens=1000)
        
        # Add test messages
        session_id = "test_session"
        
        memory.add_message(session_id, "¿Qué es OAuth2?", "user")
        memory.add_message(session_id, "OAuth2 es un protocolo de autorización...", "assistant")
        memory.add_message(session_id, "¿Cómo funciona PKCE?", "user")
        memory.add_message(session_id, "PKCE es una extensión de OAuth2...", "assistant")
        
        # Get context
        context = memory.get_context(session_id)
        print(f"Context length: {len(context)} characters")
        print(f"Context preview: {context[:200]}...")
        
        # Get stats
        stats = memory.get_session_stats(session_id)
        print(f"Session stats: {stats}")
        
        memory.shutdown()

if __name__ == "__main__":
    main()
