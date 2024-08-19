import os
import base64
import gc
import tempfile
import uuid

import pandas as pd
import streamlit as st

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, Document
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

session_id = st.session_state.id

def reset_chat():
    """Reset the chat history and context."""
    st.session_state.messages = []
    st.session_state.context = None

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
    """Read the file and return its content as a string."""
    file_type = file_path.split('.')[-1]
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'xlsx':
        df = pd.read_excel(file_path)
    else:
        return None

    # Drop rows where all elements are NaN
    df.dropna(how='all', inplace=True)

    # Convert DataFrame to a string
    content = df.to_string(index=False)
    return content

def create_documents(files):
    """Create Document objects ensuring they have content."""
    documents = []
    for file in files:
        content = read_file(file)
        if content and content.strip():
            documents.append(Document(content=content, doc_id=os.path.basename(file)))
    return documents

def retrieve_documents(query_engine, query, top_k=3):
    """Retrieve top-k relevant documents based on the query."""
    response = query_engine.query(query)
    
    # Debugging output
    print(f"Query: {query}")
    print(f"Response: {response}")
    
    documents = []
    try:
        for i, doc_response in enumerate(response):
            if i >= top_k:
                break
            documents.append(doc_response["content"])
    except Exception as e:
        print(f"Error while iterating over response: {e}")
        raise

    return documents

with st.sidebar:
    API_KEY = "nkBaDhoReFHy9tsZ9QyowMIE6qhtBLirGNN5GhXc"
    
    # File uploader widget (allow multiple files)
    uploaded_files = st.file_uploader("Choose your files", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)

    if uploaded_files and API_KEY:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []

                # Save the uploaded files temporarily
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    file_paths.append(file_path)

                file_key = f"{session_id}-{'-'.join([os.path.basename(f) for f in file_paths])}"
                st.write("Indexing your documents...")

                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                        docs = []
                        for file_path in file_paths:
                            if file_path.endswith(".pdf"):
                                # Process PDF files
                                loader = SimpleDirectoryReader(
                                    input_dir=temp_dir,
                                    required_exts=[".pdf"],
                                    recursive=True
                                )
                                docs.extend(loader.load_data())
                                display_file(file_path)
                            elif file_path.endswith(".csv") or file_path.endswith(".xlsx"):
                                # Process CSV/XLSX files
                                docs.extend(create_documents([file_path]))
                                display_file(file_path)
                        if not docs:
                            st.error('None of the uploaded files have valid content to index. Please upload valid files.')
                            st.stop()

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
                        st.session_state.query_engine = index.as_query_engine(streaming=True, node_postprocessors=[cohere_rerank])

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

                        st.session_state.query_engine.update_prompts(
                            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                        )
                        
                        st.session_state.file_cache[file_key] = st.session_state.query_engine
                    else:
                        st.session_state.query_engine = st.session_state.file_cache[file_key]

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
        
        # Retrieve relevant documents
        if st.session_state.query_engine:
            retrieved_docs = retrieve_documents(st.session_state.query_engine, prompt, top_k=3)
            
            # Combine retrieved documents for context
            context = "\n\n".join(retrieved_docs)
            
            # Simulate stream of response with milliseconds delay
            full_response = st.session_state.query_engine.query(prompt, context=context)
            
            for chunk in full_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
        else:
            st.error("Query engine is not initialized. Please upload documents first.")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})