#!/usr/bin/env python3
"""
MeLi IAM Challenge
"""
import subprocess
import time
import os
import sys
import requests
from pathlib import Path

# Add src to path para importar módulos internos
sys.path.append(str(Path(__file__).parent / "src"))

def setup_database_if_needed():
    """Verificar y configurar base de datos automáticamente si es necesario."""
    print("🔍 Verificando estado de la base de datos...")
    
    # Verificar si ChromaDB ya tiene datos
    chroma_path = Path("data/chroma")
    has_data = False
    
    if chroma_path.exists():
        # Verificar si hay archivos de base de datos
        db_files = list(chroma_path.rglob("*.bin"))
        has_data = len(db_files) > 0
    
    if has_data:
        print(" Base de datos ya configurada")
        return True
    
    print(" Base de datos vacía - configurando automáticamente...")
    
    # Ejecutar setup de base de datos
    venv_python = "./venv/bin/python"
    if not Path(venv_python).exists():
        print("  Error: venv no encontrado")
        return False
    
    try:
        print("   ⏳ Procesando documentos IAM...")
        result = subprocess.run([
            venv_python, "setup_database.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(" Base de datos configurada correctamente")
            return True
        else:
            print(f" Warning: Error en setup_database: {result.stderr}")
            print("  Continuando en modo offline...")
            return False
            
    except Exception as e:
        print(f" Warning: No se pudo configurar base de datos: {e}")
        print("  Continuando en modo offline...")
        return False

def check_ollama():
    """Verificar Ollama."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=3)
        if response.status_code == 200:
            print("Ollama funcionando correctamente")
            return True
    except:
        pass
    print("Ollama no disponible - funcionará en modo offline")
    return False

def start_api_background():
    """Iniciar API en background si es posible."""
    try:
        env = os.environ.copy()
        env['LLM_MODEL'] = 'llama3:latest'
        env['PYTHONPATH'] = '.'
        
        # Usar Python del venv específicamente
        venv_python = "./venv/bin/python"
        if not Path(venv_python).exists():
            print("Venv no encontrado - modo offline activado")
            return None
        
        print("🔧 Iniciando API FastAPI en background...")
        # Intentar iniciar API en background
        api_process = subprocess.Popen([
            venv_python, "-m", "uvicorn",
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info"  # Más verbose para debug
        ], env=env)
        
        print(" Esperando que la API esté lista...")
        time.sleep(5)  # Más tiempo para inicializar
        
        # Verificar si está funcionando
        for attempt in range(3):
            try:
                response = requests.get('http://localhost:8000/health', timeout=3)
                if response.status_code == 200:
                    print(" API funcionando en http://localhost:8000")
                    return api_process
                time.sleep(2)
            except:
                if attempt < 2:
                    print(f"Reintentando conexión API ({attempt + 1}/3)...")
                    time.sleep(2)
        
        print(" API no respondió - continuando en modo offline")
        return api_process  # Devolver proceso aunque no responda
        
    except Exception as e:
        print(f"Error iniciando API: {e}")
        return None

def main():
    """Función principal - Setup y ejecución completa."""
    print("🏆 MeLi IAM Challenge - Launcher Completo")
    print("=" * 55)
    print("CONFIGURACIÓN + EJECUCIÓN AUTOMÁTICA:")
    print("Verifica/configura base de datos")
    print("Inicia API Backend") 
    print("Inicia Frontend")
    print()
    
    # Verificar entorno
    if not Path('src/frontend/meli_app.py').exists():
        return
    
    # Verificar Ollama
    check_ollama()
    
    # Configurar base de datos automáticamente
    setup_database_if_needed()
    
    # Limpiar procesos anteriores
    try:
        subprocess.run(['pkill', '-f', 'streamlit'], capture_output=True)
        subprocess.run(['pkill', '-f', 'uvicorn'], capture_output=True)
        time.sleep(1)
    except:
        pass
    
    # Intentar iniciar API en background
    print("Iniciando API Backend...")
    api_process = start_api_background()
    
    # Iniciar frontend principal
    print(f"\n Iniciando Frontend...")
    print()
    print(" URLs DISPONIBLES:")
    if api_process:
        print(" API Backend:  http://localhost:8000")
        print(" API Docs:     http://localhost:8000/docs")
    print(" Frontend:     http://localhost:8503")
    print()
    print("Presiona Ctrl+C para detener AMBOS servicios")
    print("=" * 50)
    
    try:
        # Iniciar Streamlit con la app elegante usando venv
        env = os.environ.copy()
        env['PYTHONPATH'] = '.'
        
        venv_python = "./venv/bin/python"
        print(f"🎨 Iniciando Frontend con {venv_python}...")
        
        subprocess.run([
            venv_python, "-m", "streamlit", "run",
            "src/frontend/meli_app.py", 
            "--server.port", "8503",
            "--server.headless", "true",
            "--theme.primaryColor", "#3483fa",
            "--theme.backgroundColor", "#ffffff", 
            "--theme.secondaryBackgroundColor", "#e8f4fd",
            "--theme.textColor", "#333333"
        ], env=env)
        
    except KeyboardInterrupt:
        print(f"\n Deteniendo AMBOS servicios...")
        
    finally:
        # Limpiar API background si existe
        if api_process:
            try:
                print("   🔧 Cerrando API Backend...")
                api_process.terminate()
                time.sleep(2)
                if api_process.poll() is None:
                    api_process.kill()
                print("API Backend detenida")
            except:
                print(" API Backend ya estaba cerrada")
        
        print("Frontend detenido")

if __name__ == "__main__":
    main()
