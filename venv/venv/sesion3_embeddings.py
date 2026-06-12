# sesion3_embeddings.py
# Objetivo: Convertir texto humano a vectores matemáticos multidimensionales

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

print("🚀 1. Descargando/Cargando el cerebro matemático (Embeddings)...")
# La primera vez que lo corran, descargará ~80MB de internet. Luego será instantáneo.
modelo_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ ¡Modelo cargado correctamente!\n")

# --- EXPERIMENTO 1: ¿Cómo ve la IA una simple frase? ---
print("🔬 EXPERIMENTO 1: Traducción de texto a matemáticas")
frase_prueba = "Los profesores tienen vacaciones"
vector = modelo_embeddings.embed_query(frase_prueba)

# len() nos dirá cuántas coordenadas (dimensiones) tiene este universo
print(f"La frase '{frase_prueba}' se convirtió en un vector de {len(vector)} dimensiones.")
print("Primeros 5 números de ese vector:")
print(vector[:5]) # Solo imprimimos 5 para no saturar la pantalla
print("-" * 50)


# --- EXPERIMENTO 2: Integración con nuestros Chunks de la Sesión 2 ---
print("\n📚 EXPERIMENTO 2: Procesando nuestro Reglamento")
ruta_pdf = "Reglamento_Laboratorio_IA_v2.pdf"

try:
    # A. Ingesta (Sesión 1)
    loader = PyPDFLoader(ruta_pdf)
    documento = loader.load()

    # B. Fragmentación (Sesión 2)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    fragmentos = splitter.split_documents(documento)
    print(f"✅ Se generaron {len(fragmentos)} bloques.")

    # C. Vectorización (Sesión 3)
    # Extraemos el texto del PRIMER bloque nada más para hacer la prueba
    texto_bloque_1 = fragmentos[0].page_content
    
    print("\nCalculando el embedding del Bloque 1...")
    vector_bloque_1 = modelo_embeddings.embed_query(texto_bloque_1)
    
    print("¡Éxito! El Bloque 1 ya no es texto, ahora es pura matemática espacial.")
    print(f"Dimensiones del bloque 1: {len(vector_bloque_1)}")
    
except FileNotFoundError:
    print(f"❌ ERROR: No se encontró '{ruta_pdf}'.")
except Exception as e:
    print(f"❌ Ocurrió un error: {e}")