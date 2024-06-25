import os
import shutil
import time
from PyPDF2 import PdfReader
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    AsyncChromiumLoader
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.prompts import MessagesPlaceholder

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "faiss_index")
LLM_MODEL: str = "/mnt/backupnas/fgarcia/mistral"
VISION_MODEL: str = "/mnt/backupnas/fgarcia/llava"

def filtrar(texto,  string_a_eliminar):
    # Elimina las palabras indicadas
    texto = texto.replace("[INST]", "").replace("[/INST]", "")
    
    # Elimina la string recibida como parámetro
    texto = texto.replace(string_a_eliminar, "")
    
    return texto

def borrar_contenido_carpeta(ruta_carpeta):
    # Comprobamos si la ruta es una carpeta
    if not os.path.isdir(ruta_carpeta):
        print(f"{ruta_carpeta} no es una carpeta válida.")
        return
    
    # Obtenemos la lista de archivos en la carpeta
    lista_archivos = os.listdir(ruta_carpeta)
    
    # Eliminamos cada archivo o subcarpeta dentro de la carpeta
    for archivo in lista_archivos:
        ruta_completa = os.path.join(ruta_carpeta, archivo)
        try:
            if os.path.isfile(ruta_completa):
                os.remove(ruta_completa)
            elif os.path.isdir(ruta_completa):
                shutil.rmtree(ruta_completa)
            print(f"{ruta_completa} eliminado correctamente.")
        except Exception as e:
            print(f"No se pudo eliminar {ruta_completa}: {e}")

pdf_file = open('auxilios_supervivencia.pdf', 'rb')
text=""
if pdf_file != None:
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

chunk_size = 1000
chunk_overlap = 100

borrar_contenido_carpeta(DB_DIR)

# Chunk text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = text_splitter.split_text(text)
print("Creando la base de datos...")

db = FAISS.from_texts(chunks, embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

db.save_local("faiss_index")  
print("Base de datos creada con éxito.")
