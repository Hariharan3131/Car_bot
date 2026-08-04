from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import CarRAG

rag = CarRAG()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading car data and creating vector store...")
    try:
        documents = rag.load_data(r"D:\Gen_ai\Car\Data")
        rag.create_vectorstore(documents)
        print("✅ RAG system is ready!")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
    yield


app = FastAPI(
    title="Indian Car Consultant",
    description="RAG based Car Recommendation Chatbot",
    version="0.1",
    lifespan=lifespan
)

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
def home():
    return {"message": "Car Consultant API is running!", "status": "active"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if rag.vectorstore is not None else "not_ready",
        "vectorstore_loaded": rag.vectorstore is not None
    }


@app.post("/ask")
def ask_question(request: QueryRequest):
    if rag.vectorstore is None:
        return {"error": "RAG system not ready yet"}

    try:
        result = rag.generate_answer(request.question, k=4)
    except Exception as e:
        return {"error": f"Failed to generate answer: {e}"}

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }