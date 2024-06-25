import os
import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain_community.document_transformers import Html2TextTransformer
import requests
import time
from PIL import Image
SERVER_URL_ASK = "http://127.0.0.1:3000/pdf"

global image

# Get UI vars
############################################################

# Set the webpage title
st.set_page_config(
    page_title="First Aid Bot"
)

st.page_link("client_app.py", label="FAID", icon="⚕️")

st.title("First Aid Bot: FAID 🤖⚕️")

system_prompt = "Eres experto en medicina y primeros auxilios, se te van a realizar preguntas sobre este ámbito las cuales deberás responder haciendo uso en todo momento del contexto que se te proporciona."

image = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

############################################################

# Messaging processing
############################################################

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola!, soy FAIB, tu asistente médico. ¿Qué consulta necesitas realizar el día de hoy?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Escribe aquí", key="user_input"):

    # Add our input to the session state
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    # Add our input to the chat window
    with st.chat_message("user"):
        st.markdown(user_prompt)

############################################################
    

# Define los parámetros del modelo
def ask_user():

    global image

    if user_prompt:
        #Call the Flask API endpoint to get the answer
        if image is not None:
            print("Imagen subida.")
            files = {'file': image}
            data = {'user_prompt': user_prompt}
            response = requests.post(SERVER_URL_ASK, files=files, data=data)
        else:    
            response = requests.post(SERVER_URL_ASK, json={"user_prompt": user_prompt})

        answer = response.json().get("answer", "")

        # Add the response to the session state
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Add the response to the chat window
        with st.chat_message("assistant"):
            st.markdown(answer)

# Main function
if __name__ == '__main__':
    ask_user()
