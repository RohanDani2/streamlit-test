import os
import base64
import gc
import tempfile
import uuid
import logging

import pandas as pd
from dotenv import load_dotenv

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, Document
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

def reset_chat():
    """Reset the chat history and context."""
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def convert_excel_to_csv(excel_file_path):
    """Convert Excel file to CSV."""
    try:
        df = pd.read_excel(excel_file_path)
        csv_file_path = excel_file_path.rsplit('.', 1)[0] + '.csv'
        df.to_csv(csv_file_path, index=False)
        logger.info(f"Converted {excel_file_path} to {csv_file_path}")
        return csv_file_path
    except Exception as e:
        logger.error(f"Error converting Excel to CSV: {str(e)}")
        return None

def read_file(file_path):
    """Read and preprocess file content."""
    file_type = file_path.split('.')[-1].lower()
    if file_type in ['csv', 'xlsx']:
        try:
            df = pd.read_csv(file_path)
            df.dropna(how='all', inplace=True)
            if df.empty:
                logger.warning(f"File {file_path} is empty after dropping NaN rows.")
                return None
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return None
    return None

def create_documents(content, file_name):
    """Create document objects from file content."""
    documents = []
    if content and isinstance(content, list):
        for row in content:
            row_content = ' '.join(f"{k}: {v}" for k, v in row.items() if pd.notna(v))
            if row_content.strip():
                documents.append(Document(text=row_content, doc_id=file_name))
    if not documents:
        logger.warning(f"No valid documents created from {file_name}")
    return documents

def display_file(file_path):
    """Display the content of the file in the Streamlit app."""
    file_type = file_path.split('.')[-1].lower()
    if file_type == 'pdf':
        with open(file_path, "rb") as file:
            st.markdown(f"### PDF Preview: {os.path.basename(file_path)}")
            base64_pdf = base64.b64encode(file.read()).decode("utf-8")
            pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="300" type="application/pdf"></iframe>"""
            st.markdown(pdf_display, unsafe_allow_html=True)
    elif file_type in ['csv', 'xlsx']:
        df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)
        st.markdown(f"### File Preview: {os.path.basename(file_path)}")
        st.write(df.head())

def process_file(file, temp_dir, progress_bar):
    file_path = os.path.join(temp_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getvalue())
    
    file_type = file.name.split('.')[-1].lower()
    if file_type == 'pdf':
        loader = SimpleDirectoryReader(input_dir=temp_dir, required_exts=[".pdf"], recursive=True)
        docs = loader.load_data()
    elif file_type == 'xlsx':
        csv_file_path = convert_excel_to_csv(file_path)
        if csv_file_path:
            content = read_file(csv_file_path)
            if content:
                docs = create_documents(content, file.name)
            else:
                logger.warning(f"No valid content found in converted CSV file from {file.name}")
                return None
        else:
            logger.error(f"Failed to convert Excel file {file.name} to CSV")
            return None
    elif file_type == 'csv':
        content = read_file(file_path)
        if content:
            docs = create_documents(content, file.name)
        else:
            logger.warning(f"No valid content found in {file.name}")
            return None
    else:
        logger.warning(f'Unsupported file type: {file_type}')
        return None
    
    if not docs:
        logger.warning(f'No documents created from {file.name}')
        return None
    
    display_file(file_path)
    progress_bar.progress(30)
    return docs

# Main Streamlit app
st.title("Syncquire - Chat with Docs")

with st.sidebar:
    API_KEY = os.getenv("COHERE_API_KEY")
    
    uploaded_files = st.file_uploader("Choose your files", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)

    if uploaded_files:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                all_docs = []
                for uploaded_file in uploaded_files:
                    file_key = f"{session_id}-{uploaded_file.name}"
                    st.write(f"Processing {uploaded_file.name}...")
                    progress_bar = st.progress(0)

                    if file_key not in st.session_state.get('file_cache', {}):
                        docs = process_file(uploaded_file, temp_dir, progress_bar)
                        if docs:
                            all_docs.extend(docs)
                            st.success(f"Successfully processed {uploaded_file.name}")
                        else:
                            st.warning(f"Unable to process {uploaded_file.name}. Please check the file content.")
                    else:
                        st.write(f"{uploaded_file.name} already processed.")
                
                if all_docs:
                    llm = Cohere(api_key=API_KEY, model="command-r-plus")
                    embed_model = CohereEmbedding(
                        cohere_api_key=API_KEY,
                        model_name="embed-english-v3.0",
                        input_type="search_query",
                    )
                    cohere_rerank = CohereRerank(
                        model='rerank-english-v3.0',
                        api_key=API_KEY,
                    )

                    Settings.embed_model = embed_model
                    index = VectorStoreIndex.from_documents(all_docs, show_progress=True)
                    progress_bar.progress(60)

                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True, node_postprocessors=[cohere_rerank])
                    progress_bar.progress(90)

                    qa_prompt_tmpl_str = (
                    "You are an expert in financial analysis, particularly in mergers and acquisitions (M&A). Below is some context information from financial documents followed by a question. "
                    "Your task is to carefully analyze the context and provide a detailed, well-reasoned answer that directly reflects and is supported by the financial information in the documents. "
                    "Make sure to cite specific financial figures, metrics, and other relevant details from the context to back up your answer, and if relevant, elaborate on those details.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Based on the financial context provided, thoroughly address the following query, ensuring that your response is rooted in the information from the documents. "
                    "If any part of the question cannot be answered using the financial documents, acknowledge this and provide your best estimate or say 'I don't know'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )
                    
                    st.session_state.file_cache['combined_index'] = query_engine
                    progress_bar.progress(100)
                    st.success("All files processed. Ready to Chat!")
                else:
                    st.error("No valid content found in any of the files. Please check the files and try again.")
                    logger.error("No documents were created from any of the uploaded files.")
                    st.stop()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.exception("An error occurred during file processing")
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header("Chat with your documents")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know about the documents?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if 'combined_index' in st.session_state.file_cache:
            streaming_response = st.session_state.file_cache['combined_index'].query(prompt)
            
            for chunk in streaming_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
        else:
            st.error("No documents have been processed yet. Please upload and process files before chatting.")

    st.session_state.messages.append({"role": "assistant", "content": full_response})