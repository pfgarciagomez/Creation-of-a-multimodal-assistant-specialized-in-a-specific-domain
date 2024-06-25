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
import io
import requests
from flask import Flask, request, jsonify

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "faiss_index")
LLM_MODEL: str = "/mnt/backupnas/fgarcia/mistral"
VISION_MODEL: str = "/mnt/backupnas/fgarcia/llava"

app = Flask(__name__)

def filtrar(texto,  string_a_eliminar):
    # Elimina las palabras indicadas
    texto = texto.replace("[INST]", "").replace("[/INST]", "")
    
    # Elimina la string recibida como parámetro
    texto = texto.replace(string_a_eliminar, "")
    
    return texto

def create_chain(system_prompt, max_new_tokens, repetition_penalty, temperature, image_context):
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
    ### [INST] Instrucción:{}
    
    Contexto:{}.{}

    ### PREGUNTA:
    {}[/INST]
    """.format(system_prompt, image_context, "{context}","{question}")
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
    retriever = db.as_retriever(search_kwargs={"k": 1}) #search_kwargs={"k": 3}
    
    print(retriever)

    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )

    print("RAG chain creada.")
    return rag_chain

def visual(image):

    #Fetch the question from the user's request
    user_prompt = "Describe detallada y brevemente la imagen que se te proporciona, en castellano y en un único párrafo."

    processor = LlavaNextProcessor.from_pretrained(VISION_MODEL, local_files_only=True)

    model = LlavaNextForConditionalGeneration.from_pretrained(VISION_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True, device_map='auto',offload_folder="save_folder", load_in_4bit=True) 

    # prepare image and text prompt, using the appropriate prompt template
    prompt = "[INST] <image>\n" + user_prompt + " [/INST]"

    inputs = processor(prompt, image, return_tensors="pt").to(model.device)

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=150)

    print(str(filtrar(processor.decode(output[0], skip_special_tokens=True), user_prompt)))
    print("Imagen procesada.")
    # Return response
    return str(filtrar(processor.decode(output[0], skip_special_tokens=True), user_prompt))

# Exposing the endpoint 
@app.route('/pdf', methods=['POST'])
def pdf():
    #Fetch the question from the user's request
    temperature = 0.3
    repetition_penalty = 1.1
    max_new_tokens = 350
    system_prompt = "Responde en Castellano y en un único párrafo haciendo uso en todo momento del contexto que se te proporciona."
    tiempo_inicio = time.time()
    #Image processing:
    image_context = ""
    if 'file' in request.files:
        print("Procesando imagen...")
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        image_context = visual(image)
        user_prompt = request.form.get('user_prompt', '')
    else:
        user_prompt = request.json.get('user_prompt')
 

    llm_chain = create_chain(system_prompt, max_new_tokens, repetition_penalty, temperature, image_context)
    print("Realizando la inferencia al LLM...")
    response = llm_chain.invoke(user_prompt)
    tiempo_final = time.time()
    print(tiempo_final-tiempo_inicio)
    print(response['text'])
    # Return response
    return jsonify({"answer": response['text']})

# Run the Flask app
if __name__ =='__main__':
    app.run(port=3000)
