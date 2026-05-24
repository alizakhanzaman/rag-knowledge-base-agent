3
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Validate env ────────────────────────────────────────────────────────────
required = ["GROQ_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
missing = [k for k in required if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Missing required environment variables: {missing}")

GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME       = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME")
TAVILY_API_KEY       = os.getenv("TAVILY_API_KEY", "")
NAMESPACE            = "speech_paralinguistics"   # pdf name

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent

# ── LLM & Embeddings ────────────────────────────────────────────────────────
llm = Groq(model=LLM_MODEL_NAME, api_key=GROQ_API_KEY)
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
Settings.llm = llm
Settings.embed_model = embed_model

# ── Pinecone ──────────────────────────────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=NAMESPACE)
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(similarity_top_k=3)

# ── Tool Log Setup (Lab Task 2) ───────────────────────────────────────────────
LOG_FILE = "tool_log.txt"

def log_tool_call(tool_name: str, query: str, success: bool):
    status = "success" if success else "failed"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {tool_name} | query='{query}' | {status}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)

# ── Tool 1: Knowledge Base Search ────────────────────────────────────────────
def search_knowledge_base(query: str) -> str:
    """
    Search the internal knowledge base (SpeechParaling-Bench paper stored in Pinecone).
    Use this tool when the user asks about paralinguistics, Large Audio Language Models,
    LALMs, benchmark tasks, pairwise evaluation, Paralanguage Control, Dynamic Variation,
    Situational Adaptation, SpeechParaling-Bench, or any topic from the ingested domain PDF.
    Always prefer this tool over Wikipedia for domain-specific queries.
    Returns relevant text chunks with page citations.
    Do NOT use this tool for math calculations or recent news events.
    """
    try:
        response = query_engine.query(query)
        result = str(response)

        # Extract source nodes with page numbers
        citations = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                metadata = node.node.metadata or {}
                page = metadata.get('page_label') or metadata.get('page') or metadata.get('page_number') or '?'
                score = round(node.score, 3) if node.score else 'N/A'
                citations.append(f"[Page {page}, score={score}]")

        if citations:
            result += "\n\nSources: " + " | ".join(citations)

        log_tool_call("search_knowledge_base", query, True)
        return result
    except Exception as e:
        log_tool_call("search_knowledge_base", query, False)
        return f"Knowledge base error: {e}"

# ── Tool 2: Web Search ────────────────────────────────────────────────────────
def search_web(query: str) -> str:
    """
    Search the live internet for current or recent information.
    Use this tool when the user asks about recent events, news, trends, or anything
    that is unlikely to be in the internal knowledge base documents.
    Do NOT use this tool for questions about the uploaded PDF documents,
    math calculations, or Wikipedia-style factual overviews.
    """
    if not TAVILY_API_KEY:
        return "Web search is not configured. Please set TAVILY_API_KEY in .env"
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query, max_results=3)
        output = "\n\n".join(
            f"Source: {r['url']}\n{r['content']}"
            for r in results.get("results", [])
        )
        log_tool_call("search_web", query, True)
        return output or "No web results found."
    except Exception as e:
        log_tool_call("search_web", query, False)
        return f"Web search error: {e}"

# ── Tool 3: Calculator ────────────────────────────────────────────────────────
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the numeric result.
    Use this tool ONLY for arithmetic, algebra, or any math calculation
    (e.g., '2 ** 16', '5 * 365', 'sqrt(144)').
    Do NOT use this tool for text questions, document searches, or web lookups.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {"__import__": None})
        log_tool_call("calculate", expression, True)
        return f"Result: {result}"
    except Exception as e:
        log_tool_call("calculate", expression, False)
        return f"Calculation error: {e}"

# ── Tool 4: Wikipedia Summary (Lab Task 1) ────────────────────────────────────
def get_wikipedia_summary(topic: str) -> str:
    """
    Fetch the opening summary of a Wikipedia article for a given topic.
    Use this tool when the user asks for a factual overview, definition, or
    background on a well-known concept, technology, person, or event —
    such as 'Transformer architecture', 'Elon Musk', or 'Python programming language'.
    Do NOT use this tool for questions about uploaded PDF documents,
    math calculations, or recent news events (use search_web for those).
    """
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            summary = data.get("extract", "No summary found.")
            log_tool_call("get_wikipedia_summary", topic, True)
            return summary
        else:
            log_tool_call("get_wikipedia_summary", topic, False)
            return f"Wikipedia page not found for topic: {topic}"
    except Exception as e:
        log_tool_call("get_wikipedia_summary", topic, False)
        return f"Wikipedia error: {e}"

# ── Register Tools with ReActAgent ────────────────────────────────────────────
kb_tool   = FunctionTool.from_defaults(fn=search_knowledge_base)
web_tool  = FunctionTool.from_defaults(fn=search_web)
calc_tool = FunctionTool.from_defaults(fn=calculate)
wiki_tool = FunctionTool.from_defaults(fn=get_wikipedia_summary)

agent = ReActAgent(
    tools=[kb_tool, web_tool, calc_tool, wiki_tool],
    llm=llm,
    verbose=True,
    max_iterations=8,
    context=(
        "You are a helpful research assistant. You have four tools available: "
        "search_knowledge_base (for PDF documents), search_web (for current internet info), "
        "calculate (for math), and get_wikipedia_summary (for factual overviews). "
        "Always pick the most appropriate tool. For simple factual questions you already know, "
        "answer directly without calling any tool."
    ),
)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Agentic RAG — Lab 06")

# Allow ALL origins so the HTML file works from any port or when opened directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"endpoints": ["/health", "/tools", "/ingest", "/query", "/chat"]}

@app.get("/chat")
def serve_chat():
    return FileResponse("rag-chatbot.html")

@app.get("/health")
def health():
    stats = pinecone_index.describe_index_stats()
    vector_count = stats.get("total_vector_count", 0)
    return {
        "status": "ok",
        "pinecone_vectors": vector_count,
        "llm_model": LLM_MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "web_search_enabled": bool(TAVILY_API_KEY),
    }

@app.get("/tools")
def list_tools():
    return {
        "tools": [
            {"name": "search_knowledge_base", "description": "Search PDF knowledge base in Pinecone"},
            {"name": "search_web",             "description": "Live internet search via Tavily"},
            {"name": "calculate",              "description": "Evaluate math expressions"},
            {"name": "get_wikipedia_summary",  "description": "Fetch Wikipedia article summary"},
        ]
    }

@app.post("/ingest")
def ingest_pdf():
    import glob
    pdfs = glob.glob("*.pdf")
    if not pdfs:
        return {"status": "error", "message": "No PDF found in current directory."}
    pdf_path = pdfs[0]
    docs = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        show_progress=True,
    )
    return {
        "status": "success",
        "chunks_ingested": len(docs),
        "pdf_used": pdf_path,
    }

@app.post("/query")
async def query_agent(request: QueryRequest):
    try:
        handler = agent.run(user_msg=request.question)
        tool_trace = []

        async for event in handler.stream_events():
            event_type = type(event).__name__
            if event_type == "ToolCallResult":
                tool_trace.append({
                    "tool": str(event.tool_name),
                    "input": str(event.tool_kwargs),
                    "output_preview": str(event.tool_output)[:300] + "...",
                })

        response = await handler

        return {
            "answer": str(response),
            "tool_trace": tool_trace,
            "tools_used": list(set(t["tool"] for t in tool_trace)),
        }
    except Exception as e:
        return {"answer": f"Agent error: {str(e)}", "tool_trace": [], "tools_used": []}