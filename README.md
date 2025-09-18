# 🚀 GUÍA DE EJECUCIÓN - MELI IAM CHALLENGE

## 🌟 **MÉTODO 1**
```
### 🚀 **Ejecutar:**

```bash

# 1. Instalar todas las dependencias
./venv/bin/pip install -r requirements.txt

# 2. Configurar base de datos (primera vez)
./venv/bin/python setup_database.py

# 3. Ejecutar ambos servicios
./venv/bin/python start_meli_app.py
```

### 🌐 **URLs automáticas:**
- **Frontend:** http://localhost:8503
- **API:** http://localhost:8000  
- **Documentación:** http://localhost:8000/docs

### 🛑 **Detener:**
```bash
Ctrl+C
```

---

## 🐳 **MÉTODO 2: DOCKER (TAMBIÉN FÁCIL)**


### 🚀 **Ejecutar:**
```bash
# Construir y ejecutar
docker-compose up --build

# O en background
docker-compose up --build -d
```

### 🌐 **URLs disponibles:**
- **Frontend:** http://localhost:8503
- **API:** http://localhost:8000  
- **Documentación:** http://localhost:8000/docs

### 🛑 **Detener:**
```bash
# Si está en foreground
Ctrl+C

# Si está en background  
docker-compose down
```


## ⚠️ **SOLUCIÓN DE PROBLEMAS**

### 🔥 **Error: `./venv/bin/python no existe`**
```bash
# Recrear entorno virtual
rm -rf venv
python3 -m venv venv
./venv/bin/pip install streamlit requests
```

### 🔥 **Error: `ModuleNotFoundError`**
```bash
# Instalar dependencia específica
./venv/bin/pip install [nombre-modulo]

# O todas las dependencias
./venv/bin/pip install -r requirements.txt
```

### 🔥 **Error: `Port already in use`**
```bash
# Matar procesos en puertos
pkill -f streamlit
pkill -f uvicorn

# O usar otros puertos
./venv/bin/python -m streamlit run src/frontend/meli_app.py --server.port 8504
```

### 🔥 **Error: `API not responding`**
```bash

./venv/bin/python -m streamlit run src/frontend/meli_app.py --server.port 8503
```

---

# 📋 MeLi IAM Challenge - Documentación Técnica Completa

## Estructura 

📁 meli-iam-chatbot/
├── 🚀 start_meli_app.py          
├── 📋 README.md                   # Guía rápida de uso
├── ⚙️  setup_database.py          # Configuración inicial BD
├── 📦 requirements.txt            # Dependencias Python
├── 🔧 .env                        # Variables de entorno
├── 📝 config.env.template         # Template configuración
├── 🚫 .gitignore                  # Git ignore
│
├── 📁 src/ (13 archivos Python)    # CÓDIGO FUENTE 
│   ├── frontend/meli_app.py       # Frontend
│   ├── api/main.py + models.py    # FastAPI 
│   ├── bot/llm_interface.py       # Interfaz Llama 3
│   ├── rag/vector_store.py        # ChromaDB + RAG
│   ├── memory/conversation_memory.py # Memoria conversacional
│   └── utils/document_processor.py   # Procesamiento PDFs
│
├── 📁 data/                       # BASE DE DATOS + DOCUMENTOS
│   ├── chroma/                    # Vector database (ChromaDB)
│   ├── documents/                 # PDFs + documentación IAM
│   └── processed/                 # Chunks procesados
│
└── 📁 venv/                        # 🐍 Entorno virtual Python

## 🎯 Decisiones de Diseño

### Arquitectura del Sistema
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   ChromaDB      │
│   Streamlit     │◄──►│   Backend       │◄──►│   Vector Store  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Llama 3       │
                       │   via Ollama    │
                       └─────────────────┘
```

### 1. **¿Por qué FastAPI?**
- **Performance**: Asíncrono por defecto, ideal para LLM calls
- **Documentación automática**: OpenAPI/Swagger integrado
- **Type hints**: Validación automática de requests
- **Ecosystem**: Compatible con Pydantic, Uvicorn, etc.

### 2. **¿Por qué Llama 3?**
- **Open Source**: Cumple requisito de LLM abierto
- **Especialización**: Excelente para Q&A técnico
- **Local**: Sin dependencias de APIs externas
- **Customizable**: Control total sobre prompts

### 3. **¿Por qué ChromaDB?**
- **Vector Database**: Búsqueda semántica eficiente
- **Embeddings**: sentence-transformers integrado
- **Persistent**: Datos persisten entre reinicios
- **Metadata**: Filtros por documento/categoría

### 4. **¿Por qué memoria custom?**
- **Requisito**: "NO usar librerías ni LLM que tengan memoria"
- **Control total**: Buffer size, estrategias, persistence
- **Performance**: Optimizado para casos de uso específicos

### Métricas de Performance
- **Tiempo promedio de respuesta**: 15-20 segundos
- **Precisión RAG**: ~70% (mejora pendiente)
- **Memoria conversacional**: 29 sesiones activas
- **Documentos procesados**: 11 archivos IAM (7,000+ líneas)

## 🚀 Resultados Técnicos

### Arquitectura RAG
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Chunk Strategy**: Texto dividido por párrafos
- **Context Window**: 10 chunks máximo
- **Temperature**: 0.2 (balance precisión/creatividad)

### Memory Management
- **Estrategia**: Buffer con límite de mensajes
- **Persistencia**: JSON file-based
- **Session IDs**: Únicos por usuario
- **Cleanup**: Automático en shutdown

### API Endpoints
- **POST /api/chat**: Conversación principal
- **GET /health**: Health check
- **GET /docs**: Documentación automática

