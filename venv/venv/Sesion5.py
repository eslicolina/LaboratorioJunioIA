# sesion5_rag_completo.py
# Objetivo: Conectar ChromaDB con Llama 3.2 para generar respuestas conversacionales

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("🚀 INICIANDO EL CEREBRO PRINCIPAL...\n")

# --- 1. CARGAMOS LA MEMORIA Y LOS EMBEDDINGS (El Bibliotecario) ---
print("📥 1/3 Conectando con la Base de Datos Local...")
modelo_embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
carpeta_bd = "./bd_vectorial"

# OJO: Aquí NO hacemos from_documents. Solo cargamos la base de datos que ya existe
base_de_datos = Chroma(persist_directory=carpeta_bd, embedding_function=modelo_embeddings)

# Convertimos la BD en un "Buscador" (Retriever). Le pedimos que nos traiga los 2 mejores bloques
buscador = base_de_datos.as_retriever(search_kwargs={"k": 2})


# --- 2. CARGAMOS EL LLM (El Orador) ---
print("🧠 2/3 Despertando a Llama 3.2...")
llm = OllamaLLM(model="llama3.2")


# --- 3. DEFINIMOS LA PERSONALIDAD (El Prompt) ---
print("🎭 3/3 Configurando la personalidad de la IA...")
plantilla = """Eres el Asistente Oficial del Laboratorio de IA de la Universidad.
Tu trabajo es responder a las preguntas de los usuarios de forma amable, clara y profesional.

REGLA DE ORO: Responde ÚNICAMENTE basándote en la información del 'Contexto' que te proporciono abajo.
Si la respuesta a la pregunta no está en el Contexto, debes decir textualmente: "Lo siento, no tengo acceso a esa información en el reglamento actual." ¡NO INVENTES DATOS!

Contexto recuperado de la base de datos:
{context}

Pregunta del usuario: 
{input}
"""
prompt = ChatPromptTemplate.from_template(plantilla)

# Función auxiliar para juntar los textos de los bloques encontrados
def formatear_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 4. LA TUBERÍA LCEL (El RAG Completo) ---
# Conectamos: Buscador -> Prompt -> Llama 3.2 -> Salida en Texto
cadena_rag = (
    {"context": buscador | formatear_documentos, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. CHAT INTERACTIVO ---
print("\n" + "=" * 60)
print("🤖 ASISTENTE DEL LABORATORIO ACTIVADO")
print("Escribe 'salir' para apagar el sistema.")
print("=" * 60)

while True:
    pregunta = input("\n🙋‍♂️ Tú: ")
    
    if pregunta.lower().strip() == 'salir':
        print("🤖 Asistente: ¡Fue un placer ayudarte! Hasta luego.")
        break
        
    print("🤖 Asistente: ", end="", flush=True)
    
    # Aquí ocurre la magia: Le pasamos la pregunta a la tubería
    respuesta = cadena_rag.invoke(pregunta)
    print(respuesta)