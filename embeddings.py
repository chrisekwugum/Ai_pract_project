import os
from dotenv import load_dotenv  

from src.document_loaders import load_files_in_parallel # we have defined this function earlier in document_loader

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter 


# LEts load environment variables from a .env file
load_dotenv()

# Initialize OpenAI Embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Fetch the API key from environment variables
embedding = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=OPENAI_API_KEY)

# Load files (provide the actual directory path)
documents = load_files_in_parallel(directory="data/teaching_material")

# Function to split texts
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return [chunk for doc in documents for chunk in splitter.split_text(doc["text"])]

# Function to generate embeddings in batches
def embed_in_batches(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(embedding.embed_documents(batch))
    return embeddings

def generate_embeddings(documents):
    split_texts = split_documents(documents)
    embeddings = embed_in_batches(split_texts)
    return split_texts, embeddings

#generate_embeddings(documents)