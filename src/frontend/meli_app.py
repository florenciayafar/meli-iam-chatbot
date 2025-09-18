#!/usr/bin/env python3
import streamlit as st
import requests
import time
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

# Configuration
API_BASE_URL = "http://localhost:8000"
DEFAULT_SESSION_ID = "meli_session"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config with MercadoLibre colors
st.set_page_config(
    page_title="MeLi IAM Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"  # Panel desplegable
)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"{DEFAULT_SESSION_ID}_{int(time.time())}"
    
    if "conversation_stats" not in st.session_state:
        st.session_state.conversation_stats = {
            "total_messages": 0,
            "avg_response_time": 0,
            "session_start": time.time()
        }
    
    if "question_history" not in st.session_state:
        st.session_state.question_history = []
    
    # Estados para UX mejorada
    if "sending" not in st.session_state:
        st.session_state.sending = False
    
    if "user_input_text" not in st.session_state:
        st.session_state.user_input_text = ""
    
    if "should_scroll" not in st.session_state:
        st.session_state.should_scroll = False

def new_conversation():
    """Start a new conversation, resetting messages but keeping history."""
    st.session_state.messages = []
    st.session_state.session_id = f"{DEFAULT_SESSION_ID}_{int(time.time())}"
    st.session_state.conversation_stats = {
        "total_messages": 0,
        "avg_response_time": 0,
        "session_start": time.time()
    }
    st.session_state.user_input_text = ""
    st.session_state.sending = False

def auto_send_question(question: str):
    """Automatically send a question and generate response."""
    if not question or st.session_state.sending:
        return
    
    # Set sending state to prevent double clicks
    st.session_state.sending = True
    
    try:
        # Add to question history (keep last 10)
        if question not in st.session_state.question_history:
            st.session_state.question_history.insert(0, question)
            if len(st.session_state.question_history) > 10:
                st.session_state.question_history = st.session_state.question_history[:10]
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "timestamp": time.time()
        })
        
        # Generate response with streaming
        with st.spinner("🤖 Analizando documentos IAM..."):
            start_time = time.time()
            response = send_message_to_api(question)
            response_time = int((time.time() - start_time) * 1000)
            
            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["response"],
                "timestamp": time.time(),
                "metadata": response.get("metadata", {})
            })
            
            # Update stats
            stats = st.session_state.conversation_stats
            stats["total_messages"] += 2
            if stats["avg_response_time"] == 0:
                stats["avg_response_time"] = response_time
            else:
                stats["avg_response_time"] = (stats["avg_response_time"] + response_time) / 2
        
        # Clear input text after successful send
        st.session_state.user_input_text = ""
        
        # Activar auto-scroll después de enviar pregunta
        st.session_state.should_scroll = True
        
    except Exception as e:
        st.error(f"Error al procesar la pregunta: {str(e)}")
        logger.error(f"Error in auto_send_question: {e}")
    
    finally:
        # Always reset sending state
        st.session_state.sending = False

def send_message_to_api(message: str) -> Dict[str, Any]:
    """Send message to the chatbot API with optimized settings."""
    try:
        payload = {
            "message": message,
            "session_id": st.session_state.session_id,
            "include_sources": True,
            "max_context_chunks": 3,   # Optimizado para velocidad
            "temperature": 0.1         # Optimizado para consistencia y velocidad
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json=payload,
            timeout=25
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return get_offline_response(message)
            
    except Exception as e:
        logger.warning(f"API error: {e}")
        return get_offline_response(message)

def get_offline_response(message: str) -> Dict[str, Any]:
    """Generate offline response for common IAM questions."""
    message_lower = message.lower()
    
    # Respuestas básicas
    if "oauth" in message_lower or "autorización" in message_lower:
        response = """🔑 **OAuth 2.0 - Protocolo de Autorización:**

**¿Qué es OAuth 2.0?**
Protocolo estándar que permite a aplicaciones cliente obtener acceso limitado a recursos protegidos sin compartir credenciales del usuario.

**Componentes principales:**
• **Client** - Tu aplicación
• **Resource Server** - API con recursos protegidos  
• **Authorization Server** - Emite y valida tokens
• **Resource Owner** - Usuario final

*📚 Basado en documentación oficial*"""
    
    elif "autenticación" in message_lower or "autenticar" in message_lower:
        response = """🔐 **Métodos de Autenticación en IAM:**

**Multi-Factor (MFA):**
• **Apps Autenticadoras** - Google Authenticator, más seguro
• **Hardware Keys FIDO2** - Llaves físicas resistentes al phishing
• **Push Notifications** - Aprobación en dispositivos registrados

**Sin Contraseña (Passwordless):**
• **Passkeys** - WebAuthn con criptografía de clave pública
• **Certificados Digitales** - Basados en PKI empresarial

*📚 Basado en documentación oficial*"""
    
    elif "token" in message_lower:
        response = """⏰ **Gestión Segura de Tokens:**

**Mejores Prácticas:**
• **Short-Lived Access Tokens** - 15-60 minutos típicamente
• **Refresh Token Rotation** - Cambiar refresh token en cada uso
• **Real-time Revocation** - Capacidad de invalidación inmediata

**Riesgos de Tokens Largos:**
• Ventana de ataque ampliada
• Dificultad de revocación
• Problemas de cumplimiento

*📚 Basado en documentación oficial*"""
    
    elif "aaa" in message_lower:
        response = """🏗️ **AAA Framework - Base de IAM:**

**Authentication:** ¿Quién eres? - Verificar identidad
**Authorization:** ¿Qué puedes hacer? - Determinar permisos
**Accountability:** ¿Qué hiciste? - Registro y auditoría

**Principios:**
• **Least Privilege** - Mínimos permisos necesarios
• **Need-to-Know** - Acceso solo a información requerida

*📚 Basado en documentación oficial*"""
    
    else:
        response = """Puedo ayudarte con conceptos de Identity and Access Management basándome en documentación oficial. 

Haz una pregunta específica sobre autenticación, OAuth 2.0, tokens, o cualquier tema de IAM."""
    
    return {
        "response": response,
        "session_id": st.session_state.session_id,
        "metadata": {"offline_mode": True}
    }

def render_question_history():
    """Render recent question history for quick access."""
    if st.session_state.question_history:
        st.markdown("### 🕒 Historial Reciente")
        
        # Mostrar historial siempre visible
        for i, question in enumerate(st.session_state.question_history[:5]):
            if st.button(
                f"↺ {question[:45]}{'...' if len(question) > 45 else ''}",
                key=f"hist_{i}",
                help=f"Reenviar: {question}",
                use_container_width=True
            ):
                auto_send_question(question)
                st.rerun()

def render_predefined_questions():
    """Render predefined questions organized by IAM categories."""
    st.markdown("### 💡 Preguntas Frecuentes")
    
    # Challenge questions organized
    question_categories = {
        "🔐 Autenticación": [
            "¿Qué tipos de métodos de autenticación existen?",
            "¿Ventajas de la autenticación multifactor?",
        ],
        "🔑 OAuth & Protocolos": [
            "¿Cómo funciona OAuth 2.0?",
            "¿Qué es PKCE y por qué es importante?",
        ],
        "⏰ Tokens & Sesiones": [
            "¿Por qué no es recomendable tener token de sesión con fecha de expiración grande?",
            "¿Access tokens vs refresh tokens?",
        ],
        "🏗️ Fundamentos IAM": [
            "¿Qué significa AAA en seguridad?",
            "¿Autenticación vs autorización?",
        ]
    }
    
    for category, questions in question_categories.items():
        with st.expander(category, expanded=False):
            for question in questions:
                if st.button(
                    question,
                    key=f"q_{hash(question)}",
                    help="Click para enviar automáticamente",
                    use_container_width=True
                ):
                    auto_send_question(question)
                    st.rerun()

def check_api_status():
    """Check API connectivity status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            return True, response.json()
    except:
        pass
    return False, None

def main():
    """Main application function."""
    initialize_session_state()
    
    # CSS SIMPLE Y FUNCIONAL
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #51A2FF 0%, #3B7CDB 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message-user {
        background: linear-gradient(135deg, #FFF4B3 0%, #FFFFFF 100%);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        border-left: 4px solid #FFE600;
    }
    
    .chat-message-assistant {
        background: linear-gradient(135deg, #F5F5F5 0%, #FFFFFF 100%);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        border-left: 4px solid #51A2FF;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FFE600 0%, #E6CF00 100%);
        color: #333;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 230, 0, 0.5);
    }
    
    /* Centrar título del sidebar */
    .stSidebar h3 {
        text-align: center !important;
        margin-bottom: 1rem !important;
    }
    
    /* Botón Nueva Conversación - amarillo más claro */
    .stSidebar .stButton > button {
        background: linear-gradient(135deg, #FFF9C4 0%, #FFEB3B 100%) !important;
        color: #333 !important;
        border: 2px solid #FDD835 !important;
        border-radius: 15px !important;
        font-weight: 600 !important;
        padding: 0.7rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stSidebar .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(253, 216, 53, 0.4) !important;
        background: linear-gradient(135deg, #FFFDE7 0%, #FFEE58 100%) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 MeLi IAM Assistant</h1>
        <p>Asistente especializado en Identity and Access Management</p>
    </div>
    """, unsafe_allow_html=True)
    
# Conversación simple
    
    # Chat messages
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <h3>¡Hola! Soy tu Asistente IAM</h3>
            <p>Especializado en Identity and Access Management de MercadoLibre</p>
            <p>💡 Usa las preguntas frecuentes del panel lateral o escribe tu pregunta abajo</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message-user">
                    <strong>🧑‍💻 Tú:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                metadata = message.get("metadata", {})
                source_info = ""
                if metadata.get("offline_mode"):
                    source_info = " <em style='color:#999; font-size:0.85em;'>(Modo offline)</em>"
                
                # Procesar markdown a HTML manualmente
                content = message["content"]
                
                # Convertir markdown básico a HTML
                # Convertir **texto** a <strong>texto</strong>
                content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
                # Convertir *texto* a <em>texto</em> 
                content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
                # Convertir saltos de línea a <br>
                content = content.replace('\n', '<br>')
                
                # Mostrar todo junto en el contenedor
                st.markdown(f"""
                <div class="chat-message-assistant">
                    <strong>  🤖 MeLi IAM Assistant :</strong><br><br>{content}{source_info}
                </div>
                """, unsafe_allow_html=True)
    
    # Simple input form
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Tu pregunta:",
                placeholder="💬 Escribe tu pregunta sobre IAM aquí...",
                key="user_input_main"
            )
        
        with col2:
            send_text = "Enviando..." if st.session_state.sending else "Enviar 🚀"
            send_button = st.form_submit_button(
                send_text, 
                use_container_width=True,
                disabled=st.session_state.sending
            )
        
        if send_button and user_input.strip() and not st.session_state.sending:
            auto_send_question(user_input.strip())
            st.rerun()
        elif send_button and not user_input.strip():
            st.warning("⚠️ Por favor escribe una pregunta antes de enviar.")

    # Sidebar
    with st.sidebar:
        st.markdown("### 🤖 MeLi IAM")
        
        if st.button("➕ Nueva conversación", key="new_conv", use_container_width=True):
            new_conversation()
            # Scroll hacia arriba para mostrar el estado limpio
            st.markdown("""
            <script>
            setTimeout(function() {
                window.scrollTo({
                    top: 0,
                    behavior: 'smooth'
                });
            }, 100);
            </script>
            """, unsafe_allow_html=True)
            st.rerun()
        
        st.markdown("---")
        
        render_question_history()
        render_predefined_questions()
        
        with st.expander("⚙️ Estado del Sistema", expanded=False):
            api_online, _ = check_api_status()
            status = "✅ API Conectada" if api_online else "🔄 Modo Offline"
            st.write(status)
            
            stats = st.session_state.conversation_stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mensajes", stats["total_messages"])
            with col2:
                avg_time = stats["avg_response_time"]
                st.metric("Respuesta", f"{avg_time:.0f}ms" if avg_time > 0 else "N/A")


if __name__ == "__main__":
    main()
