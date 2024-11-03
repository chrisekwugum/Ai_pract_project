#Import relevant files
import streamlit as st
import os

import pdfplumber 
import docx
import pypandoc
from docx import Document  # For generating DOCX files
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Initialize Streamlit app
load_dotenv()
st.set_page_config(page_title="Assessment Generator", layout="wide")

# Set custom CSS for colors and layout
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar {
        background-color: #2d3e24;
        color: #e0e0e0;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        color: #90c24f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom Header
st.markdown('<div class="header"><h1>📝 Edtech Assessment Generator</h1></div>', unsafe_allow_html=True)

# API Key and Vector Store Initialization
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it in your Streamlit secrets.")
    st.stop()

# Initialize session state for message history
if "langchain_messages" not in st.session_state:
    st.session_state["langchain_messages"] = []

msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Sidebar for Customization
with st.sidebar:
    st.header("🔧 Customization Settings")
    temperature = st.slider("Response Creativity (Temperature)", 0.0, 1.0, 0.8)
    st.text("Upload as many documents as you like before generating questions.")

# Upload files
uploaded_files = st.file_uploader("Upload your teaching materials (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Save uploaded files and nake it in such a way to handle binary files(files that are not text files)
def save_uploaded_file(uploaded_file):
    upload_directory = "data/teaching_materials"
    os.makedirs(upload_directory, exist_ok=True)
    file_path = os.path.join(upload_directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

if uploaded_files:
    st.sidebar.write("Files uploaded:")
    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)
        st.sidebar.write(uploaded_file.name)

# Function to create the RAG chain

def create_rag_chain(api_key):
    try:
        embedding_function = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=api_key)
        vectorstore = Chroma(persist_directory="data/chroma_store", embedding_function=embedding_function)
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature, api_key=api_key, max_tokens=3500)
    
        # Contextualizing question based on history
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
    
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
    
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answering the user's question
        qa_system_prompt = """
        You are an expert in generating assessment questions for educators. \
        Use the following context to create comprehensive questions based on the user's input. \
        Make sure the questions are categorized and align with Bloom's Taxonomy levels as default if not specified. \
        For every assessment question generated, ensure to generate corresponding answers. \
        You could be prompted to generate multiple choice questions, use letters to number the options. \
        Make sure to format the way the question and the answer is structured. Some users can prompt you to generate the qassessment questions and answers differntly.\
        Ensure to do so, if the age range of those to be assessed is not specified, you could ask in your recommendation.\
        could also be asked to generate objective questions, ensure you keep your answers as succinct and concise as possible. \
        You could be asked to generate German questions, in the Nigerian context, these are fill-in-the-gap questions without options for answers. \
        Context: {context}"""  # Ensure that context is included
    
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
    
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
        # Combine RAG + Message History
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        return None
    
    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

rag_chain = create_rag_chain(OPENAI_API_KEY)
# Input box for user queries
if question := st.text_input("Enter your question prompt:"):
    with st.spinner("Generating assessment questions..."):
        st.chat_message("human").write(question)  # Display the user query

        # Get the response from RAG
        response = rag_chain.invoke({"input": question})
        
        # Display AI's response
        st.chat_message("ai").write(response['answer'])

        # Option to download generated questions
        if st.button("Download Generated Questions"):
            generated_content = response['answer']

            # Generate and download a PDF file
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, generated_content)
            pdf_file_path = "generated_questions.pdf"
            pdf.output(pdf_file_path)
            with open(pdf_file_path, "rb") as f:
                st.download_button(label="Download PDF", data=f, file_name=pdf_file_path)

            # Generate and download a DOCX file
            doc = Document()
            doc.add_paragraph(generated_content)
            docx_file_path = "generated_questions.docx"
            doc.save(docx_file_path)
            with open(docx_file_path, "rb") as f:
                st.download_button(label="Download DOCX", data=f, file_name=docx_file_path)

# Feedback section
st.sidebar.header("💬 Feedback")
feedback = st.text_area("Share your feedback or suggestions:")
if st.button("Submit Feedback"):
    if feedback:
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Please enter your feedback before submitting.")

# Footer
st.markdown('<div class="footer">Made with ❤️ for Educators</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">One day every Nigerian child will have access to quality education</div>')
