from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def create_vector_store():

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = Chroma(
        collection_name="tasks",
        embedding_function=embedding_model,
        persist_directory="./db/chroma_db"
    )

    return vectorstore