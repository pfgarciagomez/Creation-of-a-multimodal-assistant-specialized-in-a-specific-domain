import os
import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain_community.document_transformers import Html2TextTransformer

SERVER_URL_ASK = "http://127.0.0.1:3000/pdf"

# Get UI vars
############################################################

# Set the webpage title
st.set_page_config(
    page_title="Open Bots"
)

st.page_link("T-Bot.py", label="T-Bot", icon="📚")
st.page_link("pages/V-Bot.py", label="V-Bot", icon="🖼️")
st.page_link("pages/S-Bot.py", label="S-Bot", icon="🌐")

st.title("Teachable Bot: T-Bot 🤖⚡")

max_new_tokens = st.sidebar.slider('Max_new_tokens', 25, 500, 250)
repetition_penalty = st.sidebar.slider('Repetition_penalty', 0.0, 2.0, 1.1)
temperature = st.sidebar.slider('Temperature', 0.0, 1.0, 0.25)
chunk_size = st.sidebar.slider('Chunk_size', 500, 2000, 1000)
chunk_overlap = st.sidebar.slider('Chunk_overlap', 0, 200, 100)

system_prompt = st.text_area(
    label="Establece el Rol de T-Bot",
    value="",
    key="system_prompt")

pdf = st.file_uploader("Proporciona el contexto", type='pdf')


############################################################

# Messaging processing
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

    if user_prompt:
        #Call the Flask API endpoint to get the answer
        text=""
        if pdf != None:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

        response = requests.post(
            SERVER_URL_ASK,
            json={"user_prompt": user_prompt, "system_prompt":system_prompt, "text":text, "max_new_tokens":max_new_tokens, "repetition_penalty":repetition_penalty, "temperature":temperature, "chunk_size":chunk_size, "chunk_overlap":chunk_overlap}
        )

        answer = response.json().get("answer", "")
        context1 = response.json().get("context1", "")
        context2 = response.json().get("context2", "")
        context3 = response.json().get("context3", "")

        # Add the response to the session state
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Add the response to the chat window
        with st.chat_message("assistant"):
            st.markdown(answer)
            if text!="":
                st.markdown("CONTEXTO UTILIZADO:")
                st.markdown(context1)
                st.markdown(context2)

# Main function
if __name__ == '__main__':
    ask_user()
