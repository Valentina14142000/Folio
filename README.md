# 🧠 AI Knowledge Base

[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat&logo=next.js&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional AI-driven platform for intelligent document analysis, built with a modern **Retrieval-Augmented Generation (RAG)** architecture. It empowers users to upload documents, perform semantic search, and chat contextually with their knowledge base.

---

## ✨ Features

* **Intelligent RAG Engine:** Powered by LangChain and OpenAI for accurate, context-aware document QA.
* **FastAPI Backend:** Robust, asynchronous API designed for high-throughput document processing and embedding.
* **Modern Next.js Frontend:** Responsive, clean UI built with Next.js 15, TypeScript, and Tailwind CSS.
* **Vector Storage:** Utilizes ChromaDB for efficient local vector indexing and lightning-fast similarity retrieval.

---

## 🛠️ Tech Stack

### Backend
* **Language & Runtime:** Python 3.13
* **API Framework:** FastAPI & Uvicorn
* **Orchestration & AI:** LangChain, OpenAI API
* **Vector Database:** ChromaDB

### Frontend
* **Framework:** Next.js 15 (React)
* **Language:** TypeScript
* **Styling:** Tailwind CSS
* **HTTP Client:** Axios

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone GitHub repo
cd ai-knowledge-base
```

### 2. Set up the Backend

Navigate to the backend directory, set up your Python virtual environment, and install dependencies:

```Bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn langchain langchain-openai chromadb python-dotenv
```

Create a .env file in the backend directory and add your OpenAI API key:

```Code snippet
OPENAI_API_KEY=your_openai_api_key_here
```

Start the FastAPI server:

```Bash
uvicorn main:app --reload --port 8000
```

### 3. Set up the Frontend

Open a new terminal window, navigate to the frontend directory, and install dependencies:

```Bash
cd frontend
npm install
# or
yarn install
```

Create a .env.local file in the frontend directory if needed to configure your API endpoint:

```Code snippet
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Start the Next.js development server:

```Bash
npm run dev
# or
yarn dev
```

Open http://localhost:3000 in your browser to interact with your AI Knowledge Base dashboard.
