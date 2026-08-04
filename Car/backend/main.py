from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/")
def home():
    return {
        "message": "Car Consultant API is running successfully!",
        "status": "active"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}