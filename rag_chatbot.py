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

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
# The folder name where our vector database (FAISS) will be saved locally
INDEX_DIR = "faiss_index"


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def apply_custom_styles():
    """
    Loads and applies the custom CSS from style.css.
    Streamlit allows us to inject custom CSS directly into the web page to 
    change the background, colors, and button styles for a more premium look.
    """
    # Set the wide layout and the browser tab title
    st.set_page_config(page_title="Cynthia's RAG Q&A Hub", layout="wide", page_icon="✨")
    
    try:
        # Read the CSS file and inject it into the Streamlit app
        with open("style.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # If style.css doesn't exist yet, just ignore and keep the default look
        pass


def get_vectorstore(embeddings):
    """
    Tries to load an existing FAISS database from the local hard drive.
    FAISS is a highly efficient database designed to store and search through 
    number vectors (embeddings) instead of normal text.
    """
    if os.path.exists(INDEX_DIR):
        try:
            # We must specify allow_dangerous_deserialization=True because 
            # loading local pickle files in Python inherently assumes you trust them.
            return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.sidebar.error(f"Failed to load FAISS index: {e}")
            return None
    
    # Return None if the folder doesn't exist yet (meaning no documents were uploaded ever)
    return None


def process_uploaded_file(file):
    """
    Extracts raw text from a single uploaded file.
    Since Streamlit keeps uploaded files in RAM memory, we must momentarily 
    save them to a temporary file path so that LangChain's loaders can read them properly.
    """
    file_path = Path(f"/tmp/{file.name}")
    try:
        # Write the uploaded file from RAM to the temporary hard drive path
        file_path.write_bytes(file.getbuffer())
        
        # Pick the right "loader" depending on the file extension
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(str(file_path))
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(str(file_path))
        elif file.name.endswith(".txt"):
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            st.sidebar.warning(f"Skipping unsupported file: {file.name}")
            return []
            
        # Extract the text and return it as a list of LangChain "Document" objects
        docs = loader.load()
        return docs
    except Exception as e:
        st.sidebar.error(f"Error processing {file.name}: {e}")
        return []
    finally:
        # ALWAYS clean up the temporary file immediately after reading!
        if file_path.exists():
            file_path.unlink()


def update_vectorstore(documents, embeddings):
    """
    Takes large extracted texts, cuts them into readable 'chunks', 
    turns them into vectors (numbers), and saves them to the database.
    """
    if not documents:
        st.sidebar.error("No valid documents loaded to index.")
        return False
        
    # We can't feed an entire 100-page PDF to the AI at once.
    # So, we split it into chunks of 1000 characters.
    # The 'chunk_overlap' ensures sentences aren't randomly cut in half blindly.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    if not splits:
        st.sidebar.error("No valid document chunks created. Content might be too sparse.")
        return False

    try:
        if st.session_state.vectorstore is None:
            # Create a brand new database from the document chunks
            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
        else:
            # Or add the new documents to an existing database!
            st.session_state.vectorstore.add_documents(splits)

        # Save to local folder immediately so they stay when you refresh
        st.session_state.vectorstore.save_local(INDEX_DIR)
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to create/update FAISS index: {e}")
        st.session_state.vectorstore = None
        return False


def generate_rag_answer(question, vectorstore, llm):
    """
    Constructs the actual logic pipeline:
    1. Search the database for text chunks similar to the user's question.
    2. Feed those chunks into a magic prompt template.
    3. Send the prompt to OpenAI to get human-like answers.
    """
    # This turns the database into a "retriever" that returns the top 4 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # We instruct the AI carefully: Answer the question ONLY using the context we provide!
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on this context:\n{context}\n\nQuestion: {question}"
    )
    
    # LangChain's "LCEL" syntax: Link components together using the pipe `|` operator
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser() # Ensure the raw AI output is converted to standard text
    )
    
    # Execute the chain!
    return chain.invoke(question)


# -----------------------------------------------------------------------------
# MAIN APPLICATION LOOP
# -----------------------------------------------------------------------------
def main():
    apply_custom_styles()
    
    # 1. Sidebar Configuration UI
    st.sidebar.title("Setup")
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    uploaded_files = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT files", accept_multiple_files=True)
    
    # Reset button: Deletes the local FAISS folder entirely
    if st.sidebar.button("Reset FAISS Index"):
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
        st.session_state.vectorstore = None 
        st.sidebar.success("FAISS index reset. Upload files to rebuild.")

    # 2. Main App Configuration
    st.title("Cynthia's RAG Q&A App")

    # Stop the app completely if there is no API key
    if not api_key:
        st.warning("Add your OpenAI API key to get started.")
        st.stop()
        
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Initialize Core AI Components (Translates words -> numbers, and Brain)
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0) # temperature=0 means "be factual, no hallucinating"

    # 3. Load Existing Database 
    # Streamlit re-runs the entire script on every click. 
    # session_state helps us remember data between clicks.
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = get_vectorstore(embeddings)
        if st.session_state.vectorstore:
            st.sidebar.success("Loaded existing FAISS index.")

    # 4. Handle Incoming User Documents
    if uploaded_files:
        all_documents = []
        total_text = ""
        
        for file in uploaded_files:
            docs = process_uploaded_file(file)
            if docs:
                file_text = "\n".join(doc.page_content for doc in docs)
                total_text += file_text
                
                if file_text.strip():
                    all_documents.extend(docs)
                    st.sidebar.info(f"Loaded {file.name}: {len(file_text)} chars")
                else:
                    st.sidebar.warning(f"No text extracted from {file.name}.")
                    
        # Update database with new documents
        if all_documents and len(total_text.strip()) >= 50:
            if update_vectorstore(all_documents, embeddings):
                st.sidebar.success(f"Added {len(all_documents)} documents and saved index.")
        elif all_documents:
            st.sidebar.error("Loaded content is too short for meaningful chunks.")

        # Optional debug viewer in sidebar
        with st.sidebar.expander("Debug Loaded Content"):
            st.text(total_text if total_text else "No text extracted.")

    # 5. Render Chat History UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Draw everything that has been said previously
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 6. Handle New Chat Input
    if st.session_state.get("vectorstore"):
        # The form prevents the app from running until the user presses "Send"
        with st.form(key="query_form"):
            question = st.text_input("Your question:", key="user_input")
            submit = st.form_submit_button("Send")
            
            if submit and question:
                # Add the user's question to chat history and draw it on the screen
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                
                # Show a spinning icon while OpenAI thinks
                with st.spinner("Thinking..."):
                    try:
                        answer = generate_rag_answer(question, st.session_state.vectorstore, llm)
                    except Exception as e:
                        answer = f"Error generating answer: {e}"
                
                # Add the AI's answer to the history and draw it on the screen
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
    else:
        st.info("Upload valid files to build or load the knowledge base.")

# Standard Python boilerplate to run the app
if __name__ == "__main__":
    main()