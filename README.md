# Sistema de Preguntas y Respuestas sobre PDFs con Groq y Ollama

Este proyecto es un sistema de preguntas y respuestas sobre documentos PDF que utiliza el modelo de lenguaje de Groq y los embeddings de Ollama para procesar documentos PDF y responder preguntas basadas en su contenido.

## Características

- Carga y procesamiento de múltiples archivos PDF
- Extracción de texto de PDFs y división en fragmentos manejables
- Creación de embeddings utilizando el modelo nomic-embed-text de Ollama
- Almacenamiento de embeddings en una base de datos vectorial Chroma
- Uso del LLM de Groq para responder preguntas
- Interfaz de chat interactiva utilizando Chainlit

## Requisitos previos

Antes de ejecutar este proyecto, asegúrate de tener instalado lo siguiente:

- Python 3
- Ollama
- Modelo nomic-embed-text para Ollama


1. Instalar Ollama 


2. Descargar el modelo nomic-embed-text


3. Configura tus variables de entorno

4. Crear un archivo .env y definir ahi el GROQ_API_KEY

## Para correr
- Para correr el apppp.py: chainlit run apppp.py 
- Para correr el app_local (hacer ollama pull nomic-embed-text y ollama pull llama3)

