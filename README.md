# **CliniChat – AI Medical Assistant (RAG + Ollama + Streamlit)**

CliniChat is a lightweight AI medical assistant powered by a local RAG pipeline using **Ollama (Llama 3.2)**, **FastAPI**, **ChromaDB**, and a modern **Streamlit UI**.
The system runs entirely on your local machine, supports PDF ingestion, retrieval, chat interaction, and includes a clean medical interface suitable for academic demonstrations.

---

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Tech Stack](#2-tech-stack)  
3. [Installation](#3-installation)  
    - [3.1 Clone the Repository](#31-clone-the-repository)  
    - [3.2 Create Virtual Environment](#32-create-virtual-environment)  
    - [3.3 Install Dependencies](#33-install-dependencies)  
    - [3.4 Install Ollama and Model](#34-install-ollama-and-model)  
4. [Running the System](#4-running-the-system)  
    - [4.1 Start Backend (FastAPI)](#41-start-backend-fastapi)  
    - [4.2 Start Frontend (Streamlit UI)](#42-start-frontend-streamlit-ui)  
5. [Features](#5-features)  
6. [Acknowledgements](#6-acknowledgements)


---

## **1. Introduction**

CliniChat is a local AI assistant designed for medical question-answering using Retrieval-Augmented Generation (RAG).
It allows you to:

* Run a **local medical chatbot**
* Upload and ingest your own **medical PDF documents**
* Retrieve context using **ChromaDB**
* Get answers grounded in your documents
* View PDFs, citations, latency, and history
* Enjoy a clean UI inspired by modern medical AI tools
* Use the system completely **offline**, without API keys

Ideal for classroom demonstrations, student projects, and offline AI experimentation.

---

## **2. Tech Stack**

### **Frontend**

* Streamlit UI
* Custom CSS (CliniChat theme)
* Chat interface
* PDF viewer
* Upload panel
* History
* Hamburger menu navigation

### **Backend**

* FastAPI (`/query`, `/health`)
* RAG pipeline
* Text extraction + ingestion pipeline

### **Models & Vector Database**

* Ollama **Llama 3.2**
* **bge-small-en-v1.5** embeddings
* **ChromaDB** vector store

---

## **3. Installation**

### **3.1 Clone the Repository**

```bash
git clone https://github.com/mlinh1711/rag-medical-assistant.git
cd rag-medical-assistant
```

### **3.2 Create Virtual Environment**

```bash
python -m venv my_env
```

Activate environment (Windows):

```bash
.\my_env\Scripts\activate
```

### **3.3 Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3.4 Install Ollama and Model**

Install Ollama:
[https://ollama.com/download](https://ollama.com/download)

Pull the Llama 3.2 mini model:

```bash
ollama pull llama3.2:1b
```

---

## **4. Running the System**

Open **two terminals** in VSCode.

---

### **4.1 Start Backend (FastAPI)**

```bash
.\my_env\Scripts\activate
uvicorn backend.main:app --reload
```

Backend will be available at:

```
http://localhost:8000
http://localhost:8000/docs
```

---

### **4.2 Start Frontend (Streamlit UI)**

```bash
.\my_env\Scripts\activate
streamlit run .\frondend\streamlit_app.py
```

Frontend UI opens at:

```
http://localhost:8501
```

---

## **5. Features**

* Local medical chatbot (RAG)
* Document ingestion (PDF → text → chunks → embeddings)
* Vector retrieval using ChromaDB
* Clean chat UI with suggested questions
* PDF viewer
* Query history
* Source citations
* Latency display
* Modern interface styled after real medical AI tools

---

## **6. Acknowledgements**

* Meta Llama 3.2
* bge-small-en-v1.5 embeddings
* ChromaDB
* FastAPI
* Streamlit

---
