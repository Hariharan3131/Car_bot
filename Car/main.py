from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Car Consultant Chatbot",
    description="Indian Car Market RAG Chatbot",
    version="0.1"
)

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "🚗 Tata Car Consultant API is Running!",
        "status": "active",
        "docs": "http://127.0.0.1:8000/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}