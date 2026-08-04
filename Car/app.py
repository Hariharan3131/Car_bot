import streamlit as st
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os

st.set_page_config(page_title="Indian Car Consultant", page_icon="🚗", layout="wide")

st.title("🚗 Indian Car Consultant Chatbot")
st.write("Ask me anything about Tata cars (budget, mileage, features, comparison etc.)")

@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    csv_path = "data/tata_cleaned.csv"
    if not os.path.exists(csv_path):
        csv_path = "tata_cleaned.csv"
    
    df = pd.read_csv(csv_path)
    documents = []
    
    for _, row in df.iterrows():
        content = f"""
Brand: {row.get('brand', '')}
Model: {row.get('model', '')}
Body Type: {row.get('body_type', '')}
Fuel Type: {row.get('fuel_type', '')}
Transmission: {row.get('transmission', '')}
Engine: {row.get('engine_spec', '')}
Mileage / Range: {row.get('range_or_mileage', '')}
Price (Ex-showroom): {row.get('price_range_exshowroom_in_lakh', '')} Lakh
Seating Capacity: {row.get('seating', '')}
Boot Space: {row.get('boot_space_l', 'N/A')} liters
        """.strip()
        
        documents.append(Document(
            page_content=content,
            metadata={
                "brand": str(row.get('brand', '')),
                "model": str(row.get('model', '')),
                "fuel": str(row.get('fuel_type', ''))
            }
        ))
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectorstore

# Load the system
with st.spinner("Loading car data and building RAG system... Please wait..."):
    try:
        vectorstore = load_rag_system()
        st.success("✅ RAG system ready! You can start asking questions.")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Chat interface
question = st.text_input("Ask your question about cars:", placeholder="Example: Best car under 10 lakhs with good mileage")

if question:
    with st.spinner("Searching relevant cars..."):
        results = vectorstore.similarity_search(question, k=4)
        
        st.subheader("🔍 Relevant Cars Found:")
        
        for i, doc in enumerate(results, 1):
            with st.expander(f"Option {i}"):
                st.write(doc.page_content)