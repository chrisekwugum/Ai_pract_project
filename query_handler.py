import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import ChatOpenAI
from dotenv import load_dotenv  # Ensure environment variables are loaded

# Load environment variables from a .env file
load_dotenv()

# Setup paths
pdf_directory = "data/teaching_materials"  
chroma_db_path = "data/chroma_store"

# Ensure the ChromaDB path exists and has write permissions
if not os.path.exists(chroma_db_path):
    os.makedirs(chroma_db_path, 0o775)

# Initialize OpenAI Embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Fetch the API key from environment variables
embedding = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=OPENAI_API_KEY)

# Initialize Chroma for vector storage
chroma = Chroma(embedding_function=embedding, persist_directory=chroma_db_path)

# Function to split data into smaller batches
def batch_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Set the maximum batch size allowed by ChromaDB
MAX_BATCH_SIZE = 5461

# Function to store embeddings in ChromaDB
def store_embeddings(split_texts, embeddings, documents):
    metadata = [{'source': doc['source']} for doc in documents]
    # Split texts, embeddings, and metadata into batches
    text_batches = list(batch_data(split_texts, MAX_BATCH_SIZE))
    embedding_batches = list(batch_data(embeddings, MAX_BATCH_SIZE))
    metadata_batches = list(batch_data(metadata, MAX_BATCH_SIZE))

    # Add the batches to ChromaDB incrementally
    for text_batch, embedding_batch, metadata_batch in zip(text_batches, embedding_batches, metadata_batches):
        chroma.add_texts(texts=text_batch, embeddings=embedding_batch, metadatas=metadata_batch)

    # Persist or save with all of the metadatas to the database after all batches have been added
    chroma.persist()

# Retrieval-Augmented Generation (Retrieve + Generate)
def rag_for_assessment(question):
    # Step 1: Retrieve relevant documents from ChromaDB
    results = chroma.similarity_search(question, k=5)  

    # Step 2: Generate content with a language model (OpenAI GPT-3)
    openai_llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    retrieved_content = " ".join([result.page_content for result in results])

    # Create a prompt based on the BLOOM'S TAXONOMY framework
    prompt = f"Using the following context, generate assessment questions based on Bloom's Taxonomy:\n\n{retrieved_content}\n\nQuestion: {question}\nAnswer:"
    generated_content = openai_llm(prompt)

    return generated_content

# question = "Generate a multiple-choice question on the topic of photosynthesis."
# response = rag_for_assessment(question)
# print(response)
