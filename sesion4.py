# sesion4_chromadb.py
# Objetivo: Indexar documentos en ChromaDB y probar búsquedas semánticas interactivas

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("🚀 1. Iniciando la Tubería de Indexación (Pipeline)...")
ruta_pdf = "documento.pdf"
carpeta_bd = "./bd_vectorial"

try:
    print("📥 Leyendo el PDF...")
    loader = PyPDFLoader(ruta_pdf)
    documento = loader.load()

    print("✂️ Cortando el texto en fragmentos (Chunks)...")
    # Aplicamos un solapamiento de 100 para no perder contexto semántico
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    fragmentos = splitter.split_documents(documento)

    print("🧠 Cargando el modelo matemático Multilingüe...")
    modelo_embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    print("💾 Guardando vectores en ChromaDB (Disco Duro)...")
    # Al poner 'persist_directory', los datos sobreviven aunque cerremos Python
    base_de_datos = Chroma.from_documents(
        documents=fragmentos, 
        embedding=modelo_embeddings, 
        persist_directory=carpeta_bd
    )
    
    print(f"✅ ¡Éxito! Base de datos lista en la carpeta '{carpeta_bd}'.\n")

    # --- MOTOR DE BÚSQUEDA INTERACTIVO ---
    print("=" * 60)
    print("🔍 INICIANDO MOTOR DE BÚSQUEDA SEMÁNTICA")
    print("Escribe 'salir' en cualquier momento para apagar el sistema.")
    print("=" * 60)

    while True:
        pregunta_usuario = input("\n🙋‍♂️ Haz una pregunta al Reglamento: ")
        
        # Condición para salir del bucle
        if pregunta_usuario.lower().strip() == 'salir':
            print("👋 Apagando el motor de búsqueda. ¡Hasta la próxima!")
            break

        # k=1 significa que solo queremos que la BD nos devuelva el bloque más parecido
        resultados = base_de_datos.similarity_search(pregunta_usuario, k=1)
        
        print("\n🏆 FRAGMENTO RECUPERADO (Pura coincidencia matemática):")
        print(f"[{resultados[0].page_content}]")
        print("-" * 60)

except FileNotFoundError:
    print(f"❌ ERROR: No se encontró el archivo '{ruta_pdf}'.")
except Exception as e:
    print(f"❌ Ocurrió un error inesperado: {e}")