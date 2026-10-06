import os
import hashlib

import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# CONFIG
# ==========================================

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"   # stronger than all-MiniLM-L6-v2
LLM_MODEL = "openai/gpt-oss-120b"            # larger than gpt-oss-20b; see list_groq_models.py
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
TOP_K = 6
FETCH_K = 20

st.set_page_config(page_title="Academic Q&A", page_icon="📚", layout="centered")

st.title("📚 Academic Q&A")
st.write("Your AI academic study assistant.")


# ==========================================
# CACHED RESOURCES
# ==========================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Indexing PDF (only happens once per file)...")
def build_index(file_path: str, file_hash: str):
    """Load, split and embed the PDF. Cached by file hash, so it runs once."""
    documents = PyPDFLoader(file_path).load()

    total_chars = sum(len(d.page_content.strip()) for d in documents)
    if total_chars < 200:
        return None, len(documents), 0  # likely a scanned PDF (no text layer)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.split_documents(documents)
    docs = [d for d in docs if d.page_content.strip()]

    database = FAISS.from_documents(docs, load_embeddings())
    return database, len(documents), len(docs)


@st.cache_resource
def load_llm():
    return ChatGroq(model=LLM_MODEL, temperature=0.1)


# ==========================================
# PROMPT
# ==========================================

prompt = PromptTemplate(
    template="""You are an academic study assistant.

Answer the question using the context from the study material below.
Combine information from multiple parts of the context if needed.
Keep important details, definitions, formulas and steps from the material.
Mention page numbers when helpful.

Only if the context contains nothing relevant to the question, reply exactly:
"I don't know based on the provided study material."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

answer_chain = prompt | load_llm() | StrOutputParser()


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}" for d in docs
    )


# ==========================================
# PDF UPLOAD
# ==========================================

uploaded_file = st.file_uploader("📄 Upload your study PDF", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    os.makedirs("data/uploads", exist_ok=True)
    file_path = os.path.join("data/uploads", f"{file_hash}_{uploaded_file.name}")

    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(file_bytes)

    database, num_pages, num_chunks = build_index(file_path, file_hash)

    if database is None:
        st.error(
            "No readable text found in this PDF. It may be a scanned document "
            "and would need OCR first."
        )
        st.stop()

    st.success(f"Loaded: {uploaded_file.name} — {num_pages} pages, {num_chunks} chunks")

    retriever = database.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K},
    )

    st.divider()

    # ==========================================
    # QUESTION
    # ==========================================

    with st.form("qa_form"):
        question = st.text_input("💬 Ask a question about your PDF:")
        submitted = st.form_submit_button("🔍 Ask", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching study material and generating answer..."):
                # Retrieve ONCE, so the answer and the shown sources match
                retrieved_docs = retriever.invoke(question)
                context = format_docs(retrieved_docs)
                response = answer_chain.invoke(
                    {"context": context, "question": question}
                )

            st.subheader("🤖 Answer")
            st.write(response)

            # Sources
            st.subheader("📚 Sources")
            seen = set()
            for doc in retrieved_docs:
                source = os.path.basename(doc.metadata.get("source", "Unknown"))
                # strip the hash prefix we added to the saved filename
                source = source.split("_", 1)[-1] if "_" in source else source
                page = doc.metadata.get("page", 0) + 1
                if (source, page) not in seen:
                    st.write(f"📄 {source} — Page {page}")
                    seen.add((source, page))

            # Debug view: check if the right text is being retrieved
            with st.expander("🔎 Debug: retrieved chunks"):
                for i, d in enumerate(retrieved_docs, 1):
                    st.markdown(f"**Chunk {i} — page {d.metadata.get('page', 0) + 1}**")
                    st.text(d.page_content)


# ==========================================
# FOOTER
# ==========================================

st.divider()
st.caption("Academic Q&A • RAG-based Study Assistant")