# Medical RAG Assistant (FastAPI + Streamlit + ChromaDB + Voyage Embeddings)

Medical RAG Assistant is a lightweight question answering system powered by a Retrieval Augmented Generation pipeline.  
The system ingests medical PDF documents, builds a semantic vector store, retrieves relevant context, and generates grounded answers using an LLM through a clean and simple UI.

This project is designed for academic NLP demonstrations and fulfills the requirements of a course project on RAG based NLP systems.

---

## Table of Contents
1. Introduction  
2. Tech Stack  
3. Installation  
4. Running the System  
5. Features  
6. Evaluation Summary  
7. Acknowledgements  

---

## 1. Introduction
Medical RAG Assistant is an applied NLP system designed to answer medical questions using Retrieval Augmented Generation.

The system allows users to:
- Load medical PDF documents  
- Automatically extract and chunk the text  
- Create embeddings using a modern semantic model  
- Store vectors in ChromaDB  
- Query the system through a FastAPI backend  
- Receive grounded answers through a Streamlit front end  
- Reduce hallucination by retrieving relevant document context  

---

## 2. Tech Stack

### Frontend
- Streamlit UI  
- Clean question answer chat interface  
- Medical themed layout  
- Document ingestion panel  

### Backend
- FastAPI  
- Full RAG pipeline  
- Document chunking and preprocessing  
- Multi query retrieval support  

### Models and Vector Database
- Voyage Embeddings  
- ChromaDB vector store  

---

## 3. Installation

### 3.1 Clone the Repository
```bash
git clone https://github.com/mlinh1711/rag-medical-assistant.git
cd rag-medical-assistant
```

### 3.2 Create Virtual Environment
```bash
python -m venv .venv
```

Activate (Windows):
```bash
.\.venv\Scripts\activate
```

### 3.3 Install Dependencies
```bash
pip install -r requirements.txt
```

Set keys in `.env`:
```
OPENAI_API_KEY=your_key
VOYAGE_API_KEY=your_key
```

---

## 4. Running the System

### 4.1 Start Backend
```bash
uvicorn backend.main:app --reload
```

Backend endpoints:
```
http://localhost:8000
http://localhost:8000/docs
```

### 4.2 Start Frontend
```bash
streamlit run ./frontend/streamlit_app.py
```

Frontend UI:
```
http://localhost:8501
```

---

## 5. Features
- Medical question answering using RAG  
- PDF ingestion pipeline  
- Semantic chunking and embedding  
- Vector retrieval with ChromaDB  
- Clean chat interface  
- Query history  
- Source citations  
- Multi query retrieval  

## 7. Acknowledgements
- Deepseek and Voyage  
- ChromaDB  
- FastAPI  
- Streamlit  

---
