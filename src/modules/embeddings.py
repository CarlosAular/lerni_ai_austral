import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import ffmpeg
import io
from openai import OpenAI
import streamlit as st

load_dotenv()

os.getenv("OPENAI_API_KEY")
os.getenv("PINECONE_API_KEY")
index_name = "lerni"

def retrieveNamespace():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    namespaces = pc.Index(index_name).describe_index_stats()
    if "" in namespaces['namespaces']:
        namespaces['namespaces']["default"] = namespaces['namespaces'].pop("")
    
    
    return namespaces['namespaces'].keys()


def get_pdf_text(pdf, namespace):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks =  text_splitter.split_text(text)
    uploaded = get_vectorstore(namespace).add_texts(texts=chunks, metadatas=[{"source": pdf.name}] * len(chunks))
    print(f'Uploaded {len(uploaded)} pdf chunks')
    return uploaded

def get_video_transcription(video, namespace):
    mp4_buffer = io.BytesIO(video.getvalue())    
    process = (
        ffmpeg
        .input('pipe:0')
        .output('pipe:1', format='mp3')
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )

    out, _ = process.communicate(input=mp4_buffer.getvalue())
    get_audio_transcription(out, video.name, namespace)

def get_audio_transcription(audio, file_name, namespace):
    mp3_buffer = io.BytesIO()
    mp3_buffer.write(audio)
    mp3_buffer.seek(0)

    mp3_buffer.name = "audio.mp3"
    client = OpenAI()
    transcript = client.audio.transcriptions.create(model="whisper-1", file=mp3_buffer)

    text_splitter = CharacterTextSplitter(separator=".", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(transcript.text)
    metadata = [{"source": file_name}] * len(chunks)
    
    uploaded = get_vectorstore(namespace).add_texts(chunks, metadatas=metadata)
    print(f'Uploaded {len(uploaded)} audio chunks')

def get_vectorstore(namespace):
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
    return vectorstore
