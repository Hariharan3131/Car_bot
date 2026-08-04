from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import CarRAG
import os

app = FastAPI(
    title="Indian Car Consultant",
    description="RAG based Car Recommendation Chatbot",
    version="0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag = CarRAG()

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    print("Loading car data and creating vector store...")
    try:
        documents = rag.load_data("data/tata_cleaned.csv")
        rag.create_vectorstore(documents)
        print("✅ RAG system is ready!")
    except Exception as e:
        print(f"❌ Error loading data: {e}")

@app.get("/")
def home():
    return {
        "message": "Car Consultant API is running!",
        "status": "active"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    if rag.vectorstore is None:
        return {"error": "RAG system not ready yet"}
    
    # Retrieve relevant documents
    results = rag.vectorstore.similarity_search(request.question, k=3)
    
    context = "\n\n".join([doc.page_content for doc in results])
    
    return {
        "question": request.question,
        "relevant_cars": context,
        "message": "This is the retrieved context. We will add LLM answer next."
    }