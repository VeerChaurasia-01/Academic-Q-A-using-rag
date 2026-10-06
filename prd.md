# Product Requirements Document (PRD)

# Academic Question Answering System Using RAG

## 1. Project Overview

**Academic Q&A** is an AI-powered study assistant that allows students
to upload academic PDF documents and ask questions about their study
material.

The system uses **Retrieval-Augmented Generation (RAG)** to retrieve
relevant information from uploaded PDFs and provide answers based on
that content.

The application will use **Streamlit as the complete user interface and
application layer**. FastAPI is not used in this project.

------------------------------------------------------------------------

## 2. Problem Statement

Students often have large amounts of study material in PDF format,
making it difficult to quickly find specific information.

Traditional keyword searches may fail when a student's question uses
different wording from the document.

The proposed system allows students to ask questions naturally. It
retrieves the most relevant sections of uploaded study material and uses
an LLM to generate an answer from that retrieved context.

------------------------------------------------------------------------

## 3. Objectives

The system should:

-   Allow users to upload academic PDF files.
-   Extract text from PDFs.
-   Divide documents into smaller chunks.
-   Convert chunks into vector embeddings.
-   Store embeddings in a FAISS vector database.
-   Retrieve relevant chunks for a user's question.
-   Use an LLM to generate an answer from retrieved content.
-   Display the sources used for the answer.
-   Provide a simple, student-friendly interface.

------------------------------------------------------------------------

## 4. Target Users

Primary users include:

-   College students
-   University students
-   Engineering students
-   Students preparing for examinations
-   Students studying from PDF notes, textbooks, and lecture material

------------------------------------------------------------------------

## 5. Core Features

### 5.1 PDF Upload

Users can upload academic PDF files through the Streamlit interface.

Supported format:

``` text
.pdf
```

Example documents:

``` text
DBMS_Unit_1.pdf
Operating_System.pdf
Computer_Networks.pdf
Machine_Learning.pdf
```

### 5.2 PDF Text Extraction

The application will extract text from uploaded PDFs using
**PyPDFLoader**.

The system should preserve metadata such as:

-   File name
-   Page number

This information will later be used for source references.

### 5.3 Text Chunking

Large extracted documents will be divided into smaller chunks using
**RecursiveCharacterTextSplitter**.

Initial configuration:

``` text
Chunk size: 500
Chunk overlap: 50
```

Chunking allows the system to retrieve smaller and more relevant
sections of a document.

### 5.4 Embeddings

Each text chunk will be converted into a numerical vector using a
HuggingFace embedding model.

Initial model:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

Conceptually:

``` text
Text
  ↓
Embedding Model
  ↓
Vector
```

### 5.5 Vector Database

The generated embeddings will be stored using **FAISS**.

FAISS will allow the application to find document chunks that are
semantically similar to the user's question.

Architecture:

``` text
PDF
 ↓
Text
 ↓
Chunks
 ↓
Embeddings
 ↓
FAISS
```

### 5.6 Question Retrieval

When a user asks a question, the question will be converted into an
embedding.

The system will search FAISS for the most relevant chunks.

Initial configuration:

``` text
Top K = 3
```

This means the system retrieves the three most relevant chunks.

### 5.7 AI Answer Generation

The retrieved content will be passed to an LLM through **Groq**.

The LLM should answer using the retrieved context.

The prompt should instruct the model:

``` text
Answer only from the provided context.

If the answer is not available in the context,
say:

"I don't know based on the provided study material."
```

### 5.8 Source References

The application should display the sources used for an answer.

Example:

``` text
Sources

DBMS_Unit_1.pdf — Page 12
DBMS_Unit_1.pdf — Page 13
```

This allows students to verify the answer in their original material.

------------------------------------------------------------------------

## 6. User Interface

The application will use **Streamlit**.

Basic interface:

``` text
┌─────────────────────────────────────────────┐
│              📚 Academic Q&A                │
│                                             │
│   Your AI academic study assistant          │
│                                             │
│   📄 Upload Study Material                  │
│   [ Choose PDF ]                            │
│                                             │
│   ─────────────────────────────────────     │
│                                             │
│   💬 Ask a Question                         │
│   [ What is normalization?              ]   │
│                                             │
│              [ 🔍 Ask ]                     │
│                                             │
│   🤖 Answer                                 │
│   ─────────────────────────────────────     │
│   Normalization is...                       │
│                                             │
│   📚 Sources                                │
│   DBMS.pdf — Page 12                        │
│   DBMS.pdf — Page 13                        │
└─────────────────────────────────────────────┘
```

------------------------------------------------------------------------

## 7. Technology Stack

  Component               Technology
  ----------------------- --------------------------------
  User Interface          Streamlit
  Programming Language    Python
  PDF Processing          PyPDF / PyPDFLoader
  Text Splitting          RecursiveCharacterTextSplitter
  Embeddings              HuggingFace
  Embedding Model         all-MiniLM-L6-v2
  Vector Database         FAISS
  RAG Framework           LangChain
  LLM                     Groq
  LLM Model               Llama-compatible Groq model
  Environment Variables   python-dotenv

------------------------------------------------------------------------

## 8. System Architecture

``` text
                  ┌─────────────────┐
                  │    Streamlit    │
                  │       UI        │
                  └────────┬────────┘
                           │
                           ▼
                    Upload PDF
                           │
                           ▼
                    PyPDFLoader
                           │
                           ▼
                  Text Extraction
                           │
                           ▼
          RecursiveCharacterTextSplitter
                           │
                           ▼
                HuggingFace Embeddings
                           │
                           ▼
                       FAISS
                  Vector Database
                           │
                           │
                           │
User Question ────────────┘
       │
       ▼
Question Embedding
       │
       ▼
Similarity Search
       │
       ▼
Relevant Chunks
       │
       ▼
    LangChain
       │
       ▼
   Groq / Llama
       │
       ▼
 Answer + Sources
```

------------------------------------------------------------------------

## 9. RAG Workflow

### 9.1 Document Processing

``` text
Upload PDF
    ↓
Extract text
    ↓
Split into chunks
    ↓
Generate embeddings
    ↓
Store in FAISS
```

### 9.2 Question Answering

``` text
User asks question
        ↓
Convert question to embedding
        ↓
Search FAISS
        ↓
Retrieve relevant chunks
        ↓
Add chunks to prompt
        ↓
Send to LLM
        ↓
Generate answer
        ↓
Display answer + sources
```

------------------------------------------------------------------------

## 10. Project Folder Structure

``` text
Academic Q&A/
│
├── PRD.md
├── app.py
├── .env
├── .gitignore
│
├── data/
│   ├── uploads/
│   └── vector_store/
│
├── tests/
│
└── venv/
```

------------------------------------------------------------------------

## 11. Environment Variables

The Groq API key will be stored in `.env`.

``` text
GROQ_API_KEY=your_api_key_here
```

The API key must not be hardcoded in `app.py`.

The `.env` file should be excluded from Git using `.gitignore`.

------------------------------------------------------------------------

## 12. Functional Requirements

The application must:

1.  Allow PDF upload.
2.  Read PDF content.
3.  Preserve page information.
4.  Split extracted text into chunks.
5.  Generate embeddings.
6.  Create a FAISS vector database.
7.  Retrieve relevant chunks.
8.  Send retrieved context to the LLM.
9.  Generate an answer.
10. Display the answer.
11. Display source document and page numbers.
12. Handle questions for which relevant information cannot be found.

------------------------------------------------------------------------

## 13. Non-Functional Requirements

The application should:

-   Have a simple interface.
-   Be easy for students to use.
-   Keep API keys secure.
-   Provide reasonably fast retrieval.
-   Avoid unnecessary backend complexity.
-   Give answers grounded in uploaded material.
-   Provide source references for verification.

------------------------------------------------------------------------

## 14. Error Handling

### No PDF uploaded

``` text
Please upload a study PDF first.
```

### Empty question

``` text
Please enter a question.
```

### Missing API key

``` text
Groq API key is not configured.
```

### Answer not found

``` text
I don't know based on the provided study material.
```

### Invalid file

``` text
Please upload a valid PDF file.
```

------------------------------------------------------------------------

## 15. Future Enhancements

Possible future features:

-   Multiple PDF uploads
-   Multiple subjects
-   Persistent FAISS storage
-   Chat history
-   Follow-up questions
-   Document deletion
-   Source text preview
-   Better citation system
-   User accounts
-   Quiz generation
-   Automatic summarization
-   Flashcard generation
-   Exam question generation
-   Voice-based questions
-   Multilingual support

------------------------------------------------------------------------

## 16. MVP Scope

The first version will contain only:

``` text
PDF Upload
     ↓
Text Extraction
     ↓
Chunking
     ↓
Embeddings
     ↓
FAISS
     ↓
Retriever
     ↓
Groq LLM
     ↓
Answer
     ↓
Sources
```

There will be **no FastAPI and no separate backend server**.

Streamlit will handle the user interface, while the Python RAG pipeline
will run directly within the Streamlit application.

------------------------------------------------------------------------

## 17. Success Criteria

The MVP will be considered successful if a student can:

1.  Open the application.
2.  Upload an academic PDF.
3.  Ask a question about the PDF.
4.  Receive an answer based on the PDF.
5.  See the relevant source pages.
6.  Receive an appropriate response when the information is not present
    in the uploaded material.

------------------------------------------------------------------------

## 18. Development Steps

The project will be developed in this exact order:

``` text
Step 1  → Project setup
Step 2  → Virtual environment
Step 3  → Install libraries
Step 4  → Create folders
Step 5  → Basic Streamlit UI
Step 6  → PDF upload
Step 7  → PDF text extraction
Step 8  → Text chunking
Step 9  → Embeddings
Step 10 → FAISS
Step 11 → Retriever
Step 12 → Groq
Step 13 → RAG chain
Step 14 → Sources
Step 15 → UI improvements
```

------------------------------------------------------------------------

## 19. Final Architecture Decision

This project intentionally uses a simple architecture suitable for an
academic MVP.

``` text
Streamlit
   ↓
LangChain
   ↓
PyPDFLoader
   ↓
Text Splitter
   ↓
HuggingFace Embeddings
   ↓
FAISS
   ↓
Retriever
   ↓
Groq / Llama
   ↓
Answer + Sources
```

**FastAPI is not part of this project.**
