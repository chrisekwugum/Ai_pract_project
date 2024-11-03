## Here, we would load and process our documents, preferable in batches to help for easy processing, 
# since our application will be handling multiple files(at the same time) 

# import relevant packages
import os
import pandas as pd
import pdfplumber 
import docx
import pypandoc
from concurrent.futures import ThreadPoolExecutor, as_completed
#from langchain.document_loaders import TextLoader, CSVLoader 

# Lets seth our data directory
pdf_directory = "data/pdf_folder" 

# Define functions to handle or extract texts from pdf, txt, csv, xlsx, docx, doc files

# First function (I am using pdfplumber here instead of fitz)
def load_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Second function
def load_text(file_path):
    with open(file_path, "r") as file:
        return file.read()

# Third function (Note: I am converting DataFrame to strings for easy embedding)
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string()

# Fourth function
def load_excel(file_path):
    df = pd.read_excel(file_path)
    return df.to_string()

# Fifth function
def load_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Sixth function
def load_doc(file_path):
    """Since this is a kind of old or deprecated extension,
    it's proper to convert to plain text."""
    try:
        text = pypandoc.convert_file(file_path, "plain")
        return text
    except Exception as e:
        print(f"Error loading DOC file {file_path}: {e}")
        return ""
    
# Load and process all files in parallel to enable faster execution
def load_files_in_parallel(directory, max_workers=4):
    """Here I would be using the ThreadPoolExecutor
    for concurrent file handling as we iterate through our different file types call the functions
    defined earlier based on met conditions."""
    documents = []
    if directory is None:
        print("No directory provided.")
        return documents

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename.endswith(".pdf"):
                futures[executor.submit(load_pdf, file_path)] = filename
            elif filename.endswith(".csv"):
                futures[executor.submit(load_csv, file_path)] = filename
            elif filename.endswith(".txt"):
                futures[executor.submit(load_text, file_path)] = filename
            elif filename.endswith((".xls", ".xlsx")):  
                futures[executor.submit(load_excel, file_path)] = filename
            elif filename.endswith(".docx"):
                futures[executor.submit(load_docx, file_path)] = filename
            elif filename.endswith(".doc"):
                futures[executor.submit(load_doc, file_path)] = filename

        for future in as_completed(futures):
            filename = futures[future]
            try:
                text = future.result()
                if text:
                    documents.append({"text": text, "source": os.path.join(directory, filename)})
                else:
                    print(f"Skipped file {filename} due to extraction issues.")
            except Exception as exc:
                print(f"Exception occurred while processing {filename}: {exc}")
    
    return documents # here it goes, this returns aggregated list of all processed documments

# Load files (provide the actual directory path)
#documents = load_files_in_parallel(directory="data/teaching_material")
