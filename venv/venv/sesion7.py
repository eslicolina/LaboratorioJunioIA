# app_dinamico.py
# Objetivo: Plataforma RAG universal con carga de archivos en vivo

import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

st.set_page_config(page_title="Lector Universal IA", page_icon="📚", layout="wide")

# --- 1. BARRA LATERAL: CONTROL DE ARCHIVOS ---
with st.sidebar:
    st.header("📂 Tus Documentos")
    archivo_subido = st.file_uploader("Arrastra aquí un PDF", type="pdf")
    st.markdown("---")
    st.info("Sube un archivo para que la IA lo lea y aprenda su contenido.")

st.title("📚 Asistente de Lectura Universal")

# --- 2. EL MOTOR DE APRENDIZAJE EN VIVO ---
# Esta función solo corre cuando el usuario sube un archivo nuevo
@st.cache_resource(show_spinner=False)
def procesar_nuevo_documento(archivo):
    with st.spinner("🧠 Leyendo y memorizando el documento... (Esto puede tomar unos segundos)"):
        # A. Truco del archivo temporal (engañamos a PyPDFLoader)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(archivo.read())
            ruta_temporal = temp_file.name

        # B. Tubería clásica: Ingesta y Chunking
        loader = PyPDFLoader(ruta_temporal)
        documento = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        fragmentos = splitter.split_documents(documento)

        # C. Vectorización y Base de Datos (en memoria temporal para no mezclar archivos)
        modelo_emb = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        # Al no poner persist_directory, la BD se borra al cerrar la web, ¡Ideal para privacidad!
        bd_dinamica = Chroma.from_documents(fragmentos, modelo_emb)
        buscador = bd_dinamica.as_retriever(search_kwargs={"k": 4})
        
        # Limpiamos la basura del disco
        os.remove(ruta_temporal)
        
        return buscador

# --- 3. LÓGICA DE LA INTERFAZ CENTRAL ---
if archivo_subido is None:
    # Pantalla de espera
    st.warning("👈 Por favor, sube un documento PDF en el menú izquierdo para comenzar la charla.")
    st.session_state.mensajes = [] # Reseteamos la memoria si quitan el archivo

else:
    # Si hay archivo, encendemos el sistema
    buscador = procesar_nuevo_documento(archivo_subido)
    llm = OllamaLLM(model="llama3.2")
    
    st.success(f"¡Documento '{archivo_subido.name}' asimilado! Hazme cualquier pregunta.")

    # Memoria y chat visual (Mismo de la Sesión 6)
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["rol"]):
            st.markdown(mensaje["contenido"])

    pregunta = st.chat_input("Escribe tu pregunta...")

    if pregunta:
        # Guardamos y mostramos la pregunta
        st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        # RAG y Generación
        historial = "\n".join([f"{m['rol']}: {m['contenido']}" for m in st.session_state.mensajes[-4:]])
        docs_encontrados = buscador.invoke(pregunta)
        contexto = "\n\n".join(d.page_content for d in docs_encontrados)
        print("Radiografía del contexto")
        print(contexto)

        plantilla = f"""Eres un experto analista de documentos.
        Responde basándote ÚNICAMENTE en el Contexto provisto. 
        Si tienes la información total o general sobre lo que se pregunta, bríndala.
        Solo dí "No tengo esa información" si el tema no se menciona en lo absoluto en el document."
        
        Contexto: {contexto}
        Historial: {historial}
        Pregunta: {pregunta}
        Respuesta:"""

        with st.chat_message("assistant"):
            with st.spinner("Analizando..."):
                respuesta = llm.invoke(plantilla)
                st.markdown(respuesta)
                
        st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})