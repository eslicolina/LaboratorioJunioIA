# app.py
# Objetivo: Crear una Interfaz Web Profesional con Memoria Conversacional

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURACIÓN DE LA PÁGINA WEB ---
st.set_page_config(page_title="Asistente de Laboratorio", page_icon="🤖", layout="centered")
st.title("🤖 Asistente del Laboratorio IA")
st.markdown("¡Hola! Soy tu asistente virtual. Conozco el reglamento del laboratorio a la perfección.")

# --- 2. CARGANDO EL CEREBRO EN CACHÉ (Para no recargar en cada clic) ---
@st.cache_resource
def iniciar_sistema_ia():
    # Cargamos Embeddings y Base de Datos (Igual que en la sesión 5)
    modelo_emb = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    bd = Chroma(persist_directory="./bd_vectorial", embedding_function=modelo_emb)
    buscador = bd.as_retriever(search_kwargs={"k": 2})
    
    # Cargamos a Llama 3.2
    llm = OllamaLLM(model="llama3.2")
    return buscador, llm

buscador, llm = iniciar_sistema_ia()

# --- 3. MEMORIA CONVERSACIONAL (El Historial) ---
# Si es la primera vez que entra a la web, creamos una lista vacía de mensajes
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Dibujamos en la web todos los mensajes guardados en el historial
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

# --- 4. LA CAJA DE TEXTO DEL USUARIO ---
pregunta_usuario = st.chat_input("Escribe tu pregunta sobre el laboratorio aquí...")

if pregunta_usuario:
    # Mostramos la pregunta en la web y la guardamos en el historial
    with st.chat_message("user"):
        st.markdown(pregunta_usuario)
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta_usuario})

    # --- 5. LÓGICA DE RAG + MEMORIA ---
    # Convertimos la lista de historial en un texto continuo para que la IA lo lea
    historial_texto = "\n".join([f"{m['rol']}: {m['contenido']}" for m in st.session_state.mensajes[-4:]])
    
    # Buscamos el documento relevante
    documentos_encontrados = buscador.invoke(pregunta_usuario)
    contexto = "\n\n".join(doc.page_content for doc in documentos_encontrados)

    # El Nuevo Prompt Maestro con Memoria
    plantilla = f"""Eres el Asistente Oficial del Laboratorio de IA de la Universidad.
    Responde amablemente basándote ÚNICAMENTE en este Contexto:
    {contexto}
    
    Si la respuesta no está en el Contexto, di: "Lo siento, no tengo esa información".
    
    Aquí tienes el historial reciente de la conversación para entender el contexto:
    {historial_texto}
    
    Responde directamente al último mensaje del usuario.
    Asistente:"""

    # --- 6. LA IA RESPONDE ---
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            # Generamos respuesta y la imprimimos en la web
            respuesta_ia = llm.invoke(plantilla)
            st.markdown(respuesta_ia)
            
    # Guardamos la respuesta de la IA en la memoria
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta_ia})