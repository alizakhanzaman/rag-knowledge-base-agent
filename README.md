# RAG Knowledge Base Agent

A Retrieval-Augmented Generation (RAG) AI agent built using FastAPI, Pinecone vector database, and external knowledge tools for intelligent information retrieval and question answering.

This project demonstrates how AI agents can retrieve contextual information from vector databases and external sources before generating accurate responses.

---

## Features

- Retrieval-Augmented Generation (RAG)
- Pinecone Vector Database Integration
- FastAPI Backend
- Wikipedia Tool Integration
- Intelligent Context Retrieval
- External Knowledge Search
- REST API Endpoints
- Postman API Testing
- Tool Logging & Tracking

---

## Technologies Used

- Python
- FastAPI
- Pinecone
- Ollama / Groq
- Vector Embeddings
- REST APIs
- Postman
- JSON
- AI Agents

---

## Project Architecture

```plaintext
User Query
     ↓
FastAPI Endpoint
     ↓
Embedding Generation
     ↓
Pinecone Vector Search
     ↓
Context Retrieval
     ↓
LLM Response Generation
     ↓
Final AI Response
```

---

## Project Structure

```plaintext
rag-knowledge-base-agent/
│
├── app/
├── main.py
├── updatedmain.py
├── tool_log.py
├── .env
├── requirements.txt
├── README.md
├── workflows/
└── reports/
```

---

## Example API Endpoints

### Health Check

```http
GET /health
```

### Ask Question

```http
POST /ask
```

### Example Request

```json
{
  "query": "What is Retrieval-Augmented Generation?"
}
```

---

## Example Use Cases

- AI-powered research assistant
- Knowledge retrieval systems
- Domain-specific chatbots
- Software engineering assistants
- Intelligent document search
- FAQ automation systems

---

## Screenshots

<img width="766" height="593" alt="image" src="https://github.com/user-attachments/assets/a1bcc2af-27f8-4613-9d23-9aacd1d8ef19" />
<img width="945" height="728" alt="image" src="https://github.com/user-attachments/assets/50a98d28-7b9f-45a1-818d-48788255667d" />
<img width="813" height="607" alt="image" src="https://github.com/user-attachments/assets/61239352-4d39-4a85-907e-c2e4e624398b" />
<img width="822" height="628" alt="image" src="https://github.com/user-attachments/assets/8239c767-76d3-430e-bad2-7ebb5b76de64" />
<img width="975" height="566" alt="image" src="https://github.com/user-attachments/assets/70f69257-0031-4d4c-8c16-86c9c4f20575" />


Example screenshots to add:
- FastAPI Swagger UI
- Pinecone dashboard
- Postman testing
- Workflow execution
- API responses

---

## Learning Outcomes

- Understanding RAG Architecture
- Vector Database Integration
- Embedding-based Search
- AI Agent Development
- API Development with FastAPI
- Knowledge Retrieval Systems
- Context-Aware AI Responses

---

## Installation

### Clone Repository

```bash
git clone https://github.com/alizakhanzaman/rag-knowledge-base-agent.git
```

### Navigate to Project Folder

```bash
cd rag-knowledge-base-agent
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run FastAPI Server

```bash
uvicorn main:app --reload
```

---

## API Testing

Use:
- Postman
- FastAPI Swagger Docs
- cURL

Swagger Docs:

```plaintext
http://127.0.0.1:8000/docs
```

---

## Repository Contents

| Folder/File | Description |
|---|---|
| app/ | Application modules |
| main.py | FastAPI entry point |
| tools.py | External tools integration |
| vector_store.py | Pinecone vector logic |
| workflows/ | AI workflow files |
| screenshots/ | Project screenshots |
| reports/ | Lab reports/documentation |

---

## Academic Context

This project was developed as part of the Generative AI for Software Engineering coursework to explore Retrieval-Augmented Generation systems and AI-powered knowledge retrieval.

---

## Author

Aliza Zaman  
BS Software Engineering

---

## Status

Completed ✅
