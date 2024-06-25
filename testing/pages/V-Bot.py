import os
import streamlit as st
import nest_asyncio
import shutil
import time
import requests

from PyPDF2 import PdfReader
from langchain_community.document_loaders import (
     DirectoryLoader,
     PyPDFLoader,
     TextLoader,
     AsyncChromiumLoader)

SERVER_URL_ASK = "http://127.0.0.1:3000/visual"

# Get UI vars
############################################################

st.page_link("T-Bot.py", label="T-Bot", icon="📚")
st.page_link("pages/V-Bot.py", label="V-Bot", icon="🖼️")
st.page_link("pages/S-Bot.py", label="S-Bot", icon="🌐")

st.title("Visual Bot: V-Bot 🤖⚡")

url = st.text_area(
    label="URL de la imagen a analizar por V-Bot",
    value="https://hips.hearstapps.com/hmg-prod/images/escritorio-de-madera-con-almacenaje-1597243654.jpg",
    key="url")

user_prompt = st.sidebar.text_area(
    label="Visual Prompt",
    value="",
    key="prompt")

st.image(url)

############################################################

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¿Qué necesitas el día de hoy?"}
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

    if user_prompt and url:
        # Call the Flask API endpoint to get the answer
        response = requests.post(
            SERVER_URL_ASK,
            json={"user_prompt": user_prompt, "url":url}
        )
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