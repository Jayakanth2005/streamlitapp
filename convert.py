import streamlit as st
import cv2
import numpy as np
import PyPDF2
import tempfile
from PIL import Image, ImageDraw, ImageFont
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, LLMChain
from langchain.llms import Cohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import CohereEmbeddings
import os
from dotenv import load_dotenv
from streamlit_chat import message
from gtts import gTTS

# Load environment variables
load_dotenv()
api_key = os.getenv('COHERE_API_KEY')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

# Convert text to speech and save as an MP3 file
def text_to_audio(text):
    if len(text) > 5000:
        text = text[:5000]
    tts = gTTS(text, lang="en")
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio_file.name)
    return temp_audio_file.name

# Convert PDF text to an animated scrolling video using OpenCV
def text_to_animated_video(text):
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    width, height = 1280, 720
    fps = 30
    duration = 10  # Animation duration in seconds
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_file.name, fourcc, fps, (width, height))

    # Use default font if system font is unavailable
    try:
        font_path = r"C:\Windows\Fonts\Arial.ttf"  # Windows default Arial font
        font = ImageFont.truetype(font_path, 40)
    except IOError:
        font = ImageFont.load_default()  # Use PIL's default font if Arial is missing

    # Create a scrolling text: we split the first 100 words into lines of 10 words each
    words = text.split()
    lines = [" ".join(words[i:i+10]) for i in range(0, min(len(words), 100), 10)]
    full_text = "\n".join(lines)

    for i in range(fps * duration):
        img = Image.new("RGB", (width, height), (0, 0, 0))  # Black background
        draw = ImageDraw.Draw(img)
        # Compute vertical position: start at bottom and scroll upward
        y_position = height - int(i * (height / (fps * duration)))
        draw.text((50, y_position), full_text, font=font, fill="white")
        frame = np.array(img)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    return temp_video_file.name

# Summarize PDF text using a recursive (chunk-based) approach so that the final summary spans 4–5 pages
def summarize_text(text):
    # First split the text into manageable chunks.
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    chunks = splitter.split_text(text)
    
    summarizer = Cohere(model="command-light", cohere_api_key=api_key)
    chunk_summaries = []
    for chunk in chunks:
        prompt = (
            "Please summarize the following text into bullet points, including all key details. "
            "Ensure that the summary is detailed enough so that when all chunk summaries are combined, "
            "the final summary will span about 4 to 5 pages:\n\n" + chunk
        )
        summary_chunk = summarizer(prompt)
        chunk_summaries.append(summary_chunk)
    
    # Combine all chunk summaries
    combined_summary = "\n".join(chunk_summaries)
    final_prompt = (
        "Based on the following bullet point summaries, create a comprehensive final summary "
        "that is detailed and spans about 4 to 5 pages, covering all key details:\n\n" + combined_summary
    )
    final_summary = summarizer(final_prompt)
    return final_summary

# -----------------------------
# Main Streamlit App Interface
# -----------------------------
st.title("📄 PDF to Animated Video (Scrolling Text), Audio, Summary & Chatbot")
st.subheader("Upload your PDF file to generate an animated teaching video, narrated audio, a detailed summary, and chat with its content.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save uploaded PDF temporarily and extract text
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        pdf_text = extract_text_from_pdf(temp_pdf.name)

    st.write("Extracted text from PDF:")
    st.text_area("PDF Content", pdf_text, height=200)

    # Summarize PDF Text into bullet points spanning 4-5 pages
    summary = summarize_text(pdf_text)
    st.subheader("📌 PDF Summary (Bullet Points):")
    st.write(summary)
    st.download_button(
        label="Download Summary",
        data=summary,
        file_name="pdf_summary.txt",
        mime="text/plain"
    )

    # Convert PDF to narrated audio
    audio_file_path = text_to_audio(pdf_text)
    st.audio(audio_file_path, format="audio/mp3")
    st.download_button("Download Audio", data=open(audio_file_path, "rb"), file_name="pdf_audio.mp3", mime="audio/mp3")

    # Generate an animated scrolling video from the PDF text
    video_file_path = text_to_animated_video(pdf_text)
    if video_file_path:
        st.video(video_file_path)
        st.download_button("Download Animated Video", data=open(video_file_path, "rb"), file_name="animated_pdf_video.mp4", mime="video/mp4")
    else:
        st.error("Error generating video. Check OpenCV installation!")

    # -----------------------------
    # Set up the Chatbot Feature
    # -----------------------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = text_splitter.split_text(pdf_text)
    embeddings = CohereEmbeddings(cohere_api_key=api_key, user_agent="convert.py")
    db = FAISS.from_texts(chunks, embeddings)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "include_metadata": True})

    template = (
        "You are a job recruiter agent working for a job portal, assigned with the task of providing "
        "information on any queries of the job seeker.\n"
        "INSTRUCTIONS:\n"
        "- Always use a friendly tone in your conversation.\n"
        "- Provide responses with bullet points, paragraphs, and proper headings when necessary.\n"
        "- Always respond in English.\n"
        "- If no relevant information is available, reply with 'I am sorry, this information is not available.'"
    )
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = Cohere(cohere_api_key=api_key)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    memory = ConversationBufferMemory(output_key='answer', memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, chain_type="stuff", retriever=retriever)

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    st.subheader("Chat with the PDF")
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything from the provided PDF."]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!"]

    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Ask something about the PDF", key='input')
            submit_button = st.form_submit_button(label='Send')
        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")