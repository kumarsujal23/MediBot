from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

data_path="C:/Users/Lenovo/Desktop/medibot/data"
def load_pdf_files(data):
    loader=DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents
doc=load_pdf_files(data=data_path)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    test_chunks=text_splitter.split_documents(extracted_data)
    return test_chunks

text_chunks=create_chunks(doc)


def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
embedding_model=get_embedding_model()


db_FAISS_PATH="C:/Users/Lenovo/Desktop/medibot/vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(db_FAISS_PATH)