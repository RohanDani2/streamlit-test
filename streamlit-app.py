import os
import base64
import gc
import tempfile
import uuid

import pandas as pd

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, Document
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

import streamlit as st

# Initialize session state
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

def display_file(file_path):
    """Display the content of the file in the Streamlit app."""
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        with open(file_path, "rb") as file:
            st.markdown("### PDF Preview")
            base64_pdf = base64.b64encode(file.read()).decode("utf-8")
            pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                                style="height:100vh; width:100%">
                            </iframe>"""
            st.markdown(pdf_display, unsafe_allow_html=True)
    elif file_type == 'csv':
        df = pd.read_csv(file_path)
        st.markdown("### CSV Preview")
        st.write(df)
    elif file_type == 'xlsx':
        df = pd.read_excel(file_path)
        st.markdown("### XLSX Preview")
        st.write(df)

def read_file(file_path):
    file_type = file_path.split('.')[-1]
    if file_type in ['csv', 'xlsx']:
        df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)
        
        # Drop rows where all elements are NaN
        df.dropna(how='all', inplace=True)
        
        # Convert non-empty rows to a list of dictionaries
        non_empty_rows = df.dropna(how='all').to_dict('records')
        
        # Join non-empty rows into a string, with each row on a new line
        content = '\n'.join([str(row) for row in non_empty_rows if row])
        return content
    return None

def create_documents(content, file_name):
    documents = []
    if content and content.strip():
        documents.append(Document(text=content, doc_id=file_name))
    return documents

with st.sidebar:
    API_KEY = "nkBaDhoReFHy9tsZ9QyowMIE6qhtBLirGNN5GhXc"
    
    # File uploader widget
    uploaded_file = st.file_uploader("Choose your files", type=["pdf", "csv", "xlsx"])

    if uploaded_file and API_KEY:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                # Save the uploaded file temporarily
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                        if uploaded_file.type == "application/pdf":
                            # Process PDF files
                            loader = SimpleDirectoryReader(
                                input_dir=temp_dir,
                                required_exts=[".pdf"],
                                recursive=True
                            )
                            docs = loader.load_data()
                            display_file(file_path)
                        elif uploaded_file.type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                            # Process CSV/XLSX files
                            content = read_file(file_path)
                            docs = create_documents(content, uploaded_file.name)
                            if not docs:
                                st.error('The uploaded file has no valid content to index. Please upload a file with non-empty data.')
                                st.stop()
                            display_file(file_path)
                    else:
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()

                    if docs:
                        # Setup LLM & embedding model
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

                        # Create an index over loaded data
                        Settings.embed_model = embed_model
                        index = VectorStoreIndex.from_documents(docs, show_progress=True)

                        # Create the query engine
                        Settings.llm = llm
                        query_engine = index.as_query_engine(streaming=True, node_postprocessors=[cohere_rerank])

                        # Customize prompt template
                        qa_prompt_tmpl_str = (
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                        )
                        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                        query_engine.update_prompts(
                            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                        )
                        
                        st.session_state.file_cache[file_key] = query_engine
                        st.success("Ready to Chat!")
                    else:
                        st.error("No valid content found in the file. Please check the file and try again.")
                        st.stop()
                else:
                    query_engine = st.session_state.file_cache[file_key]
                    st.success("Ready to Chat!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()
            
col1, col2 = st.columns([6, 1])

with col1:
    st.header("Syncquire - Chat with Docs")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})