import os
import glob
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain_core.documents import Document


class CarRAG:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.persist_directory = persist_directory
        self.vectorstore = None

    def load_data(self, folder_path: str = r"D:\Gen_ai\Car\Data") -> list[Document]:
        """Load all brand CSV files from a folder and convert them into Documents."""
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: '{folder_path}'")

        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found inside '{folder_path}'")

        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                dfs.append(df)
                print(f"Loaded {len(df)} rows from {os.path.basename(file)}")
            except Exception as e:
                print(f"⚠️ Skipping {file}: {e}")

        if not dfs:
            raise ValueError("No CSV files could be loaded successfully.")

        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Total combined rows: {len(combined_df)}")

        documents = []
        for _, row in combined_df.iterrows():
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
                    "fuel": str(row.get('fuel_type', '')),
                }
            ))

        return documents

    def create_vectorstore(self, documents: list[Document]):
        """Build (or load) the Chroma vector store from documents."""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("Building new vector store...")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        return self.vectorstore

    def query(self, question: str, k: int = 5):
        """Run a similarity search against the vector store."""
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore not initialized. Call create_vectorstore() first.")
        return self.vectorstore.similarity_search(question, k=k)