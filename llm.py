import streamlit as st
import os
import speech_recognition as sr
import pyttsx3
import tempfile
import av
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
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Speech synthesis (AI Voice Response)
def speak(text):
    """Convert text to speech."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()

# Sidebar Navigation
st.sidebar.title("🗂️ Navigation")
page = st.sidebar.radio("Go to:", ["Chatbot (Text & File)", "Voice Chat"])

# 📌 **Main Page - Text & File Chatbot**
if page == "Chatbot (Text & File)":
    st.title("🧠 Smart Chatbot (PDF, Image, Text)")

    api_key = st.sidebar.text_input("🔑 Groq API Key:", type="password")
    temperature = st.sidebar.slider("🌡️ Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.sidebar.slider("✍️ Max Tokens", 50, 300, 150)

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

    # Display Chat
    st.write("💬 **Chat with me!**")
    for message in st.session_state.messages:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.write(message["content"])

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
        
        response = generate_response(user_input, context)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

# 🎙️ **Voice Chat Page**
elif page == "Voice Chat":
    st.title("🎙️ Voice Chat with AI")

    class AudioProcessor(AudioProcessorBase):
        """Process audio stream for speech recognition."""
        def recv(self, frame: av.AudioFrame):
            audio_data = frame.to_ndarray()
            recognizer = sr.Recognizer()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tmpfile.write(audio_data.tobytes())
                tmpfile_path = tmpfile.name

            try:
                with sr.AudioFile(tmpfile_path) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio)
                    st.session_state["transcribed_text"] = text
            except sr.UnknownValueError:
                st.session_state["transcribed_text"] = "🔇 Couldn't recognize speech. Try again."
            except sr.RequestError:
                st.session_state["transcribed_text"] = "❌ Speech recognition service unavailable."
            return frame

    webrtc_streamer(
        key="voice-chat",
        mode="sendonly",
        audio_processor_factory=AudioProcessor,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
    )

    if "transcribed_text" in st.session_state and st.session_state["transcribed_text"]:
        user_voice_text = st.session_state["transcribed_text"]
        st.write(f"🗣️ You said: **{user_voice_text}**")

        # **AI Response**
        response = generate_response(user_voice_text)
        st.write("🤖 AI says:")
        st.write(response)

        # **Speak Response**
        speak(response)
