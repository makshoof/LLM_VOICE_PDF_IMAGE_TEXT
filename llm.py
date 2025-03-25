import streamlit as st
import os
import speech_recognition as sr
import pyttsx3
import threading
import base64
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from PIL import Image

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Text-to-Speech (TTS) Function
def speak(text):
    """Speak text using pyttsx3 in a separate thread."""
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            print(f"🔊 Speaking: {text}")  # Debugging print
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"❌ Speech Error: {e}")

    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()

# Streamlit UI Setup
st.title("🧠 Smart Chatbot (Voice + PDF + Image + Text)")

# Sidebar Settings
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("🔑 Groq API Key:", type="password")
temperature = st.sidebar.slider("🌡️ Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("✍️ Max Tokens", 50, 300, 150)
voice_enabled = st.sidebar.checkbox("🎙️ Enable Voice Chat")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectors" not in st.session_state:
    st.session_state.vectors = None

# File Upload
uploaded_files = st.file_uploader("+", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=False)

def process_files():
    if uploaded_files:
        if uploaded_files.type == "application/pdf":
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_files.read())
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)
            st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)
            st.success("✅ PDF processed successfully!")
        else:
            image = Image.open(uploaded_files)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.session_state.image = image
            st.success("✅ Image uploaded successfully!")

process_files()

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    If there is no relevant context, provide a general response.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def generate_response(question, context=""):
    """Generate response using LLM."""
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=temperature, max_tokens=max_tokens)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        return chain.invoke({'input': question, 'context': context})
    except Exception as e:
        return f"Error: {e}"

# Voice Recording (Using JavaScript)
st.markdown("""
    <script>
        let mediaRecorder;
        let audioChunks = [];
        function startRecording() {
            navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
            });
        }
        function stopRecording() {
            mediaRecorder.stop();
            mediaRecorder.onstop = () => {
                let audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                let reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    let base64data = reader.result.split(',')[1];
                    fetch('/_st_audio_upload', {
                        method: 'POST',
                        body: JSON.stringify({ audio: base64data }),
                        headers: { 'Content-Type': 'application/json' }
                    }).then(response => response.json()).then(data => {
                        document.getElementById('transcribed_text').value = data.text;
                        document.getElementById('submit_button').click();
                    });
                };
            };
        }
    </script>
    <button onclick="startRecording()">🎙️ Start Recording</button>
    <button onclick="stopRecording()">🛑 Stop Recording</button>
    <input type="hidden" id="transcribed_text">
    <button id="submit_button" style="display:none;"></button>
""", unsafe_allow_html=True)

# API to handle audio transcription
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/_st_audio_upload', methods=['POST'])
def process_audio():
    data = request.get_json()
    audio_data = base64.b64decode(data['audio'])
    recognizer = sr.Recognizer()
    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data)
        with sr.AudioFile("temp_audio.wav") as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"text": f"Error: {e}"})

# Display Chat
st.write("💬 **Chat with me!**")
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(message["content"])

# User Input (Text)
user_input = st.chat_input("Type your message or ask about the uploaded file...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    context = ""
    if st.session_state.vectors:
        document_chain = create_stuff_documents_chain(ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile"), prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        context = retrieval_chain.invoke({'input': user_input}).get('answer', "")
        st.session_state.vectors = None  # Clear vectors after one use
    
    response = generate_response(user_input, context)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
    
    if voice_enabled:
        speak(response)  # Speak response

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.session_state.vectors = None
    if "image" in st.session_state:
        del st.session_state["image"]
    st.success("Chat history cleared!")
