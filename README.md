# 📄 AI Document Intelligence Platform

A production-ready platform for uploading, processing, and chatting with your documents. Built with FastAPI, LangGraph, LangChain, and MongoDB, this backend powers the Document Intelligence platform.

**Frontend Application:** [Frontend GitHub Repository](https://github.com/janaka99/ai-document-intelligence-platform-client) | **Live Demo:** [https://docintel-nine.vercel.app](https://docintel-nine.vercel.app)

---

## ✨ Features

- **Document Processing Pipeline:** Upload documents, extract text, and automatically chunk them for processing.
- **Vector Embeddings:** Generate and store vector embeddings in MongoDB for fast semantic search and retrieval.
- **Streaming Chat:** Chat with your documents in real-time using Server-Sent Events (SSE) and LangGraph's stateful agent graphs.
- **Authentication & User Management:** Secure JWT-based authentication using `fastapi-users` backed by SQLite/SQLAlchemy.
- **LLM Integration:** Seamless integration with OpenAI models via LangChain.
- **Rate Limiting & Token Management:** Built-in safeguards to manage conversational turns and token budgets per session.
- **Structured Logging:** JSON logs with automatic `request_id` tracing via `structlog`.

---

## 🛠️ Technology Stack

- **Framework:** FastAPI
- **AI & Agents:** LangGraph, LangChain, OpenAI
- **Database (Relational):** SQLite + SQLAlchemy (for Users, Documents Metadata, Chat Sessions)
- **Database (Vector/NoSQL):** MongoDB (for Document Chunks and Vector Embeddings)
- **Authentication:** FastAPI-Users
- **Logging:** Structlog
- **Configuration:** Pydantic Settings

---

## 📁 Project Structure

```text
ai-docuement-intelligence-platform/
├── app/
│   ├── main.py                  # FastAPI app entry point, middleware, routes
│   ├── api/                     # API routers (document, chat, agent, embedding)
│   ├── agents/                  # LangGraph agents (chat_agent, document_agent, etc.)
│   ├── core/                    # Config, logging, exception handlers, rate limiters
│   ├── db/                      # SQLAlchemy & MongoDB connection setup and queries
│   ├── models/                  # SQLAlchemy ORM models (User, Document, ChatSession)
│   ├── schemas/                 # Pydantic models for request/response validation
│   └── tools/                   # Custom tools for agents (e.g., search_documents, chunking)
├── uploads/                     # Local storage for uploaded document files
├── .env                         # Environment variables configuration
├── requirements.txt             # Python dependencies
└── pyproject.toml               # Project metadata
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- MongoDB instance (local or Atlas)
- OpenAI API Key

### 2. Clone & Create Virtual Environment

```bash
git clone <this-repository-url>
cd ai-docuement-intelligence-platform

python -m venv .venv
# Activate environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Copy the example environment file and update it with your credentials:

```bash
cp .env.example .env
```

Ensure your `.env` contains the required keys:

```env
APP_NAME="DocIntel API"
APP_ENV="development"
APP_VERSION="0.1.0"
LOG_LEVEL="INFO"

# LLM
OPENAI_API_KEY="sk-..."

# Databases
DATABASE_URL="sqlite+aiosqlite:///./app.db"
MONGO_URI="mongodb://localhost:27017/"
MONGO_DB_NAME="docintel"

# Security
SECRET_KEY="your-super-secret-key"
```

### 5. Run the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.
API Documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.

---

## 🔌 Core API Endpoints

### Authentication (`/auth` & `/users`)

- `POST /auth/jwt/login`: Authenticate and get JWT token
- `POST /auth/register`: Register a new user

### Documents (`/api/v1/document`)

- `POST /upload`: Upload a new document
- `POST /chunk/{document_id}`: Chunk document text and save to MongoDB (Streaming progress)
- `POST /train/{document_id}`: Generate vector embeddings for document chunks (Streaming progress)
- `GET /`: List all user documents

### Chat (`/api/v1/chat`)

- `POST /sessions`: Create a new chat session for a document
- `POST /sessions/{session_id}/chat`: Send a message and get a streaming response (SSE)
- `GET /sessions/{session_id}/messages`: Retrieve chat history

---

## 🧠 LangGraph Chat Architecture

The chat system utilizes **LangGraph** to manage conversational state and conditionally route tasks:

1. **User Message:** Received via the SSE chat endpoint.
2. **Agent Node:** The LLM decides whether it can answer directly or needs to search the document.
3. **Tool Node:** If a search is needed, the `search_documents` tool queries MongoDB for relevant vector embeddings.
4. **Response:** The LLM streams the synthesized answer back to the user in real-time.

## 📄 License

MIT License
