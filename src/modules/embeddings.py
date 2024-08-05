import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

os.getenv("OPENAI_API_KEY")
os.getenv("PINECONE_API_KEY")
index_name = "lerni"
namespace = "austral"


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore_with_text(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstores = PineconeVectorStore.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        index_name=index_name
    )
    return vectorstores


def get_vectorstore():
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return vectorstore
