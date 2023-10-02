import streamlit as st 
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch 
import textwrap 
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
# from pydantic_settings import BaseSettings
# from constants import CHROMA_SETTINGS
# from streamlit_chat import message

# import speech_recognition as sr
# from googletrans import Translator
# import language_tool_python
# from textblob import Word, TextBlob
# from happytransformer import HappyTextToText as HappyTTT
# from happytransformer import TTSettings

import chromadb
from chromadb.config import Settings 

#!pip install streamlit SpeechRecognition googletrans==4.0.0-rc1 language-tool-python textblob happytransformer
# pip install pyaudio

CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory='db',
        anonymized_telemetry=False
)

st.set_page_config(layout="wide")

device = torch.device('cpu')

checkpoint = "MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)


# checkpoint = "LaMini-T5-738M"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# base_model = AutoModelForSeq2SeqLM.from_pretrained(
#     checkpoint,
#     device_map="auto",
#     torch_dtype = torch.float32,
#     from_tf=True
# )

persist_directory = "db"

@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #create vector store here
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None 

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.95,
        device=device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function = embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data
#function to display the PDF of a given file 
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        # message(history["past"][i], is_user=True, key=str(i) + "_user")
        # message(history["generated"][i],key=str(i))
        st.write(history["past"][i])
        st.write(history["generated"][i])

def main():
    # st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 🦜📄 </h1>", unsafe_allow_html=True)
    # st.markdown("<h3 style='text-align: center; color: grey;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color:white;'>Omdena: UAE chapter</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color:white;'>LLM local deployment test</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = "docs/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

        col1, col2= st.columns([1,2])
        with col1:
            st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style color:black;'>File preview</h4>", unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)

        with col2:
            with st.spinner('Embeddings are in process...'):
                ingested_data = data_ingestion()
            st.success('Embeddings are created successfully!')
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)


            user_input = st.text_input("", key="input")

            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]
                
            # Search the database for a response based on user input and update session state
            if user_input:
                answer = process_answer({'query': user_input})
                st.session_state["past"].append(user_input)
                response = answer
                st.session_state["generated"].append(response)

            # Display conversation history using Streamlit messages
            if st.session_state["generated"]:
                display_conversation(st.session_state)
        
###Speeech functions
def transcribe_audio(audio_path=None, use_microphone=False):
    if audio_path:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
    elif use_microphone:
        with sr.Microphone() as source:
            st.text("Listening...")
            audio = recognizer.listen(source)
    else:
        raise ValueError("Either provide an audio file path or set use_microphone to True.")

    try:
        if audio:
            transcription = recognizer.recognize_google(audio)
            return transcription
    except sr.UnknownValueError:
        return "Unable to transcribe audio"
    except sr.RequestError:
        return "Could not request results; check your network connection"

def Grammer_Fixer(Text):
   Grammer = HappyTTT("T5","prithivida/grammar_error_correcter_v1")
   config = TTSettings(do_sample=True, top_k=10, max_length=100)
   corrected = Grammer.generate_text(Text, args=config)
   return corrected.text

def fix_paragraph_words(paragraph):
    sentence = TextBlob(paragraph)
    correction = sentence.correct()
    return correction

def fix_word_spell(word):
    word = Word(word)
    correction = word.correct()
    return correction

def correct_grammar(text):
    corrected_text = tool.correct(text)
    return corrected_text

def translate_text(text, target_language):
    translator = Translator()
    translated_text = translator.translate(str(text), dest=target_language)  # Convert TextBlob to string
    return translated_text.text
        
def speech():
    recognizer = sr.Recognizer()
    audio_option = st.radio("Select audio source:", ("Upload Audio", "Use Microphone"))
    if audio_option == "Upload Audio":
        uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            transcription = transcribe_audio(uploaded_file.name)
            # Apply grammar correction (replace Grammer_Fixer with your grammar correction logic)
            corrected_transcription = Grammer_Fixer(transcription)
            corrected_transcription = fix_paragraph_words(corrected_transcription)  # Apply spell fixing
            target_language = st.selectbox("Select Target Language:", ["en", "es", "fr"])
            translated_text = translate_text(corrected_transcription, target_language)
            st.subheader("Translated Text:")
            st.write(translated_text)
    
    elif audio_option == "Use Microphone":
        st.info("Click 'Start' to begin recording.")
        if st.button("Start"):
            transcription = transcribe_audio(use_microphone=True)
            st.subheader("Transcription:")
            st.write(transcription)
            # Apply grammar correction (replace Grammer_Fixer with your grammar correction logic)
            corrected_transcription = Grammer_Fixer(transcription)
            corrected_transcription = fix_paragraph_words(corrected_transcription)  # Apply spell fixing
            target_language = st.selectbox("Select Target Language:", ["en", "es", "fr"])
            translated_text = translate_text(corrected_transcription, target_language)
            st.subheader("Translated Text:")
            st.write(translated_text)



if __name__ == "__main__":
    # speech()
    main()


