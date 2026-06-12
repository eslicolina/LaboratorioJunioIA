# sesion2_chunking.py
# Objetivo: Aprender a dividir documentos grandes preservando el contexto semántico.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print(" 1. Cargando el documento original...")
ruta_pdf = "documento.pdf"

try:
    loader = PyPDFLoader(ruta_pdf)
    documento_completo = loader.load()
    
    # Extraemos todo el texto crudo para medirlo
    texto_total = "\n".join(doc.page_content for doc in documento_completo)
    print(f" Documento cargado. Tamaño total: {len(texto_total)} caracteres.\n")

    print(" 2. Aplicando estrategia de Chunking (Fragmentación)...")
    
    # Intenta dividir por párrafos dobles (\n\n), luego por saltos simples (\n), luego espacios.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,       # Límite de caracteres por bloque
        chunk_overlap=100,     # Caracteres que se repiten para no perder el hilo
        length_function=len   # Usamos la función len() nativa de Python para medir
    )

    # Ejecutamos la división
    fragmentos = text_splitter.split_documents(documento_completo)
    
    print(f"📊 El documento fue dividido en {len(fragmentos)} bloques (chunks).\n")

    # 3. Auditoría de Datos: Vamos a inspeccionar los primeros 3 bloques
    print("--- INSPECCIÓN DE LOS FRAGMENTOS ---")
    for i, fragmento in enumerate(fragmentos[:3]):
        print(f"\n🔹 BLOQUE {i + 1} (Tamaño: {len(fragmento.page_content)} caracteres):")
        print(f"TEXTO: {fragmento.page_content}")
        print("-" * 60)

    print("\n💡 TAREA PARA EL ALUMNO: Modifica el chunk_size a 50 y observa qué ocurre con las palabras.")

except FileNotFoundError:
    print(f"❌ ERROR: No se encontró '{ruta_pdf}'.")
except Exception as e:
    print(f"❌ Ocurrió un error: {e}")