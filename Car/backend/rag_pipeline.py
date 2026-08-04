import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class CarRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None

    def load_data(self, csv_path="data/tata_cleaned.csv"):
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
            Mileage: {row.get('range_or_mileage', '')}
            Price: {row.get('price_range_exshowroom_in_lakh', '')} Lakh
            Seating: {row.get('seating', '')}
            """
            documents.append(Document(
                page_content=content.strip(),
                metadata={
                    "brand": str(row.get('brand', '')),
                    "model": str(row.get('model', '')),
                    "fuel_type": str(row.get('fuel_type', ''))
                }
            ))

        return documents

    def create_vectorstore(self, documents):
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        print("✅ Vector store created successfully!")
        return self.vectorstore