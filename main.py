import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain



from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

import google.generativeai as genai
import os

from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------
# 0️⃣ Configure Gemini API Key
# -----------------------------------------------------
def set_gemini_key():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("❌ GOOGLE_API_KEY not found in .env file")
        return

    genai.configure(api_key=api_key)



# -----------------------------------------------------
# 1️⃣ Load and split PDFs
# -----------------------------------------------------
def load_pdfs(uploaded_files):
    import tempfile
    all_pages = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        all_pages.extend(pages)

    return all_pages


# -----------------------------------------------------
# 2️⃣ Split into chunks
# -----------------------------------------------------
def split_into_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=50
    )
    return splitter.split_documents(pages)



# -----------------------------------------------------
# 3️⃣ Create vector DB
# -----------------------------------------------------
def create_vector_db(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    db = FAISS.from_documents(docs, embeddings)
    return db


# -----------------------------------------------------
# 4️⃣ Create Gemini QA Chain
# -----------------------------------------------------

def create_qa_chain(db):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    return qa



# -----------------------------------------------------
# 5️⃣ Handle Query
# -----------------------------------------------------
def answer_query(qa, query):
    return qa(query)


# -----------------------------------------------------
# 6️⃣ Streamlit UI
# -----------------------------------------------------
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs (Gemini)")
    st.title("📚 Chat with Multiple PDFs (Gemini + LangChain)")

    set_gemini_key()

    # Track uploaded PDF names
    if "uploaded_names" not in st.session_state:
        st.session_state["uploaded_names"] = []

    # Chat history
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Upload PDFs
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    # ---------------------------------------------------------
    # If new PDFs are uploaded, reprocess and rebuild vector DB
    # ---------------------------------------------------------
    if uploaded_files:
        current_files = [file.name for file in uploaded_files]

        if current_files != st.session_state["uploaded_names"]:
            with st.spinner("Processing PDFs... ⏳"):
                pages = load_pdfs(uploaded_files)
                docs = split_into_chunks(pages)
                db = create_vector_db(docs)
                st.session_state.qa = create_qa_chain(db)

            st.session_state["uploaded_names"] = current_files
            st.success("📁 PDFs processed successfully!")

    # ---------------------------------------------------------
    # Start chat once QA chain exists
    # ---------------------------------------------------------
    if "qa" in st.session_state:

        # Show past messages
        if st.session_state["history"]:
            st.write("### 💬 Conversation")
            for user_msg, bot_msg in st.session_state["history"]:
                with st.chat_message("user"):
                    st.write(user_msg)

                with st.chat_message("assistant"):
                    st.write(bot_msg)

        # Display chat input box at bottom
        query = st.chat_input("Ask a question from the PDFs...")

        if query:
            result = st.session_state.qa({
                "question": query,
                "chat_history": st.session_state["history"]
            })

            answer = result["answer"]

            # Display latest conversation
            st.chat_message("user").write(query)
            st.chat_message("assistant").write(answer)

            # Save conversation history
            st.session_state["history"].append((query, answer))

            # Display sources
            with st.expander("📎 Sources"):
                for doc in result["source_documents"]:
                    filename = doc.metadata.get("source", "Unknown File")
                    page = doc.metadata.get("page", 0) + 1
                    st.write(f"{filename} — Page {page}")

            # Save history (not displayed per your requirement)
            #st.session_state["history"].append((query, answer))

        
if __name__ == "__main__":
    main()
