import os
import streamlit as st
import nest_asyncio
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

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from flask import Flask, request, jsonify

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "faiss_index")
LLM_MODEL: str = "/mnt/backupnas/fgarcia/mistral"
VISION_MODEL: str = "/mnt/backupnas/fgarcia/llava"

global prevText
global prevUrl
prevUrl = ""
prevText = ""

app = Flask(__name__)

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

def ingest(text, chunk_size, chunk_overlap, isURL):
    borrar_contenido_carpeta(DB_DIR)

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if isURL:
        chunks = text_splitter.split_documents(text)
    else:    
        chunks = text_splitter.split_text(text)
    st.write(chunks)
    print("Creando la base de datos...")

    # Load chunked documents into the FAISS index
    if isURL:
        db = FAISS.from_documents(documents=chunks,
                            embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    else:
        db = FAISS.from_texts(chunks, embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

    db.save_local("faiss_index")  
    print("Base de datos creada con éxito.")

def create_chain(system_prompt, max_new_tokens, repetition_penalty, temperature):
    print("Generando modelo...")
    model_path = LLM_MODEL
    model_type="mistral"

    config = {'max_new_tokens': max_new_tokens, 'repetition_penalty': repetition_penalty, 'temperature': temperature, 'context_length': 4000}

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")

    llm = CTransformers(
        model=model_path,
        model_type=model_type,
        config=config
    )
    
    print("Modelo cargado.")
    # Prompt template
    prompt_template = """
    ### [INST] Instrucción: Responde en español y únicamente a lo que se te pregunta utilizando como ayuda el contexto que se te proporciona. Responde en un único párrafo. {}

    {}

    ### PREGUNTA:
    {}[/INST]
    """.format(system_prompt,"{context}","{question}")
    # Create prompt from prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Create llm chain

    print("Creando RAG chain...")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Load Faiss vector

    # Initialize HuggingFace embeddings
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda"},
    )

    db = FAISS.load_local("faiss_index", embeddings=huggingface_embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2}) #search_kwargs={"k": 3}

    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )

    print("RAG chain creada.")
    return rag_chain

def create_chain_simple(system_prompt, max_new_tokens, repetition_penalty, temperature):

    model_path = LLM_MODEL
    model_type="mistral"

    config = {'max_new_tokens': max_new_tokens, 'repetition_penalty': repetition_penalty, 'temperature': temperature, 'context_length': 2048}

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")

    llm = CTransformers(
        model=model_path,
        model_type=model_type,
        config=config
    )

    # Prompt template

    prompt_template = """
    ### [INST] Instrucción: Responde en español y únicamente a lo que se te pregunta. Responde en un único párrafo. {}

    ### PREGUNTA:
    {}[/INST]
    """.format(system_prompt,"{question}")
    # Create prompt from prompt template
    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template,
    )

    # Create llm chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    rag_chain = (
    {"question": RunnablePassthrough()}
        | llm_chain
    )
    return llm_chain

# Exposing the endpoint 
@app.route('/pdf', methods=['POST'])
def pdf():
    global prevText

    # Fetch the question from the user's request
    temperature = request.json.get('temperature')
    repetition_penalty = request.json.get('repetition_penalty')
    chunk_size = request.json.get('chunk_size')
    chunk_overlap = request.json.get('chunk_overlap')
    max_new_tokens = request.json.get('max_new_tokens')
    system_prompt = request.json.get('system_prompt')
    user_prompt = request.json.get('user_prompt')
    text = request.json.get('text')

    if None in (temperature, repetition_penalty, chunk_size, chunk_overlap, max_new_tokens, system_prompt, user_prompt):
        return "Error: Falta algún parámetro en la petición.", 400  

    if "pdf" in request.files:
        pdf = request.files['pdf']
        print("Pdf recibido.")
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

    if text!=prevText:
        ingest(text, chunk_size, chunk_overlap, False)
        prevText = text

    if text != "":
        llm_chain = create_chain(system_prompt, max_new_tokens, repetition_penalty, temperature)
    else:
        llm_chain = create_chain_simple(system_prompt, max_new_tokens, repetition_penalty, temperature)
    tiempo_inicial = time.time()
    response = llm_chain.invoke(user_prompt)
    tiempo_final = time.time()
    print(tiempo_final-tiempo_inicial)
    print(response['text'])
    # Return response
    if text!= "":
        return jsonify({"answer": response['text'], "context1": str(response['context'][0]), "context2": str(response['context'][1])})
    else:
        return jsonify({"answer": response['text']})

# Exposing the endpoint 
@app.route('/search', methods=['POST'])
def search():
    global prevUrl

    # Fetch the question from the user's request
    temperature = request.json.get('temperature')
    repetition_penalty = request.json.get('repetition_penalty')
    chunk_size = request.json.get('chunk_size')
    chunk_overlap = request.json.get('chunk_overlap')
    max_new_tokens = request.json.get('max_new_tokens')
    system_prompt = request.json.get('system_prompt')
    url = request.json.get('url')
    user_prompt = request.json.get('user_prompt')

    if url != prevUrl:
        nest_asyncio.apply()

        # Articles to index
        articles = url.split(',')

        # Scrapes the blogs above
        loader = AsyncChromiumLoader(articles)
        docs = loader.load()

        # Converts HTML to plain text
        html2text = Html2TextTransformer()
        text = html2text.transform_documents(docs)
        ingest(text, chunk_size, chunk_overlap, True)
        prevUrl = text

    if text != "":
        llm_chain = create_chain(system_prompt, max_new_tokens, repetition_penalty, temperature)
    else:
        llm_chain = create_chain_simple(system_prompt, max_new_tokens, repetition_penalty, temperature)
    tiempo_inicial = time.time()
    response = llm_chain.invoke(user_prompt)
    tiempo_final = time.time()
    print(tiempo_final-tiempo_inicial)
    print(response['text'])
    # Return response
    return jsonify({"answer": response['text']})

# Exposing the endpoint 
@app.route('/visual', methods=['POST'])
def visual():

    # Fetch the question from the user's request
    user_prompt = request.json.get('user_prompt')
    url = request.json.get('url')

    processor = LlavaNextProcessor.from_pretrained(VISION_MODEL, local_files_only=True)

    model = LlavaNextForConditionalGeneration.from_pretrained(VISION_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True, device_map='auto',offload_folder="save_folder", load_in_4bit=True) 

    # prepare image and text prompt, using the appropriate prompt template
    image = Image.open(requests.get(url, stream=True).raw)
    prompt = "[INST] <image>\n" + user_prompt + " [/INST]"

    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    tiempo_inicial = time.time()
    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=250)
    tiempo_final = time.time()
    print(tiempo_final-tiempo_inicial)
    print(processor.decode(output[0], skip_special_tokens=True))
    
    # Return response
    return jsonify({"answer": str(filtrar(processor.decode(output[0], skip_special_tokens=True),user_prompt))})

# Run the Flask app
if __name__ =='__main__':
    app.run(port=3000)
