import os
from pathlib import Path
import shutil
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# from dotenv import load_dotenv
# load_dotenv()


# Constants
INDEX_DIR = "faiss_index"

# Sidebar for API key, file upload, and controls
st.sidebar.title("Setup")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
uploaded_files = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT files", accept_multiple_files=True)
if st.sidebar.button("Reset FAISS Index"):
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    st.session_state.vectorstore = None
    st.sidebar.success("FAISS index reset. Upload files to rebuild.")

# Main app
st.title("Cynthia's RAG Q&A App")

if not api_key:
    st.warning("Add your OpenAI API key to get started.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# Load or initialize vectorstore
if "vectorstore" not in st.session_state:
    if os.path.exists(INDEX_DIR):
        try:
            st.session_state.vectorstore = FAISS.load_local(INDEX_DIR, embeddings, 
                                                            allow_dangerous_deserialization=True)
            st.sidebar.success("Loaded existing FAISS index.")
        except Exception as e:
            st.sidebar.error(f"Failed to load FAISS index: {e}")
            st.session_state.vectorstore = None
    else:
        st.session_state.vectorstore = None

# Process uploaded files
documents = []
total_text = ""
if uploaded_files:
    for file in uploaded_files:
        file_path = Path(f"/tmp/{file.name}")
        try:
            file_path.write_bytes(file.getbuffer())
            
            if file.name.endswith(".pdf"):
                try:
                    loader = PyPDFLoader(str(file_path))
                    loaded_docs = loader.load()
                except Exception as e:
                    st.sidebar.error(f"PyPDFLoader failed for {file.name}: {e}. File may be image-based (needs OCR) or protected.")
                    loaded_docs = []
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(str(file_path))
                loaded_docs = loader.load()
            elif file.name.endswith(".txt"):
                loader = TextLoader(str(file_path), encoding="utf-8")
                loaded_docs = loader.load()
            else:
                st.sidebar.warning(f"Skipping unsupported file: {file.name}")
                continue
            
            file_text = "\n".join(doc.page_content for doc in loaded_docs) if loaded_docs else ""
            total_text += file_text
            
            if loaded_docs and file_text.strip():
                documents.extend(loaded_docs)
                st.sidebar.info(f"Loaded {file.name}: {len(file_text)} chars (preview: {file_text[:100]}...)")
            else:
                st.sidebar.warning(f"No content loaded from {file.name}. File might be empty, non-text, or image-based.")
            
            file_path.unlink()  # Clean up
        except Exception as e:
            st.sidebar.error(f"Error processing {file.name}: {e}")
            if file_path.exists():
                file_path.unlink()

    if documents:
        if len(total_text.strip()) < 50:
            st.sidebar.error("Loaded content is too short for meaningful chunks. Add files with more text.")
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)

            if splits:
                try:
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
                    else:
                        st.session_state.vectorstore.add_documents(splits)

                    # Save to local folder
                    st.session_state.vectorstore.save_local(INDEX_DIR)
                    st.sidebar.success(f"Added {len(documents)} documents and saved index.")
                except Exception as e:
                    st.sidebar.error(f"Failed to create/update FAISS index: {e}")
                    st.session_state.vectorstore = None
            else:
                st.sidebar.error("No valid document chunks created. Content might be too sparse—check previews above.")
    else:
        st.sidebar.error("No valid documents loaded from uploaded files.")

    # Debug expander for full content
    with st.sidebar.expander("Debug Loaded Content"):
        st.text(total_text if total_text else "No text extracted.")

# Chat history init
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query input with send button
if st.session_state.get("vectorstore"):
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    # RAG chain
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on this context:\n{context}\n\nQuestion: {question}"
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    with st.form(key="query_form"):
        question = st.text_input("Your question:", key="user_input")
        submit = st.form_submit_button("Send")
        
        if submit and question:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate answer
            with st.spinner("Thinking..."):
                try:
                    answer = chain.invoke(question)
                except Exception as e:
                    answer = f"Error generating answer: {e}"
            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
else:
    st.info("Upload valid files to build or load the knowledge base.")