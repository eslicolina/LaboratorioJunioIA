from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("🚀 1. Iniciando la lectura del PDF...")

# 1. Cargamos el PDF
ruta_pdf = "documento.pdf"

try:
    loader = PyPDFLoader(ruta_pdf)
    documentos = loader.load()
    print(f"✅ ¡PDF '{ruta_pdf}' cargado! ({len(documentos)} páginas leídas).\n")

    # 2. Conectamos con el "Cerebro" (nuestro modelo local)
    print("🧠 2. Despertando a Llama 3.2 en Ollama...")
    llm = OllamaLLM(model="llama3.2")

    # 3. Le damos instrucciones estrictas a la IA
    prompt = ChatPromptTemplate.from_template("""
    Eres un asistente de la Universidad. Responde a la pregunta del usuario basándote ÚNICAMENTE en el texto del documento que te proporciono abajo. 
    Si la respuesta no está en el documento, di "No tengo esa información".

    DOCUMENTO:
    {context}

    PREGUNTA DEL USUARIO: 
    {input}
    """)

    # 4. Extraemos el texto crudo de los documentos leídos
    texto_extraido = "\n\n".join(doc.page_content for doc in documentos)

    # 5. Creamos la cadena directa usando la arquitectura moderna LCEL (sin usar .chains)
    cadena_qa = prompt | llm | StrOutputParser()

    # 6. ¡Hacemos que el usuario interactúe!
    print("-" * 50)
    pregunta_usuario = input("🙋‍♂️ Escribe tu pregunta sobre el reglamento: ")
    
    print("\n🤖 IA procesando el documento y pensando su respuesta...")
    
    # Le pasamos a la IA el texto extraído y la pregunta del usuario
    respuesta = cadena_qa.invoke({
        "context": texto_extraido, 
        "input": pregunta_usuario
    })

    print("\n✨ RESPUESTA DE LA IA:")
    print(respuesta)
    print("-" * 50)

except FileNotFoundError:
    print(f"❌ ERROR: No se encontró el archivo '{ruta_pdf}'. Asegúrate de que esté en la misma carpeta.")
except Exception as e:
    print(f"❌ Ocurrió un error inesperado: {e}")