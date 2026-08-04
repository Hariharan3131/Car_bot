import os
import glob
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class CarRAG:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.persist_directory = persist_directory
        self.vectorstore = None

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=os.environ.get("GROQ_API_KEY")
        )

        self.prompt = ChatPromptTemplate.from_template("""
You are a helpful Indian car consultant. Use ONLY the car data provided below to answer the user's question.
If the data doesn't contain a good match, say so honestly instead of guessing.

Conversation so far:
{chat_history}

Car Data (relevant to the current question):
{context}

Current Question: {question}

Give a clear, helpful answer. If this question refers back to something discussed earlier (e.g. "what about a cheaper one", "compare that with..."), use the conversation history to understand what's being referred to. Recommend specific models with brief reasons, and mention price/mileage where relevant.
""")

    def load_data(self, folder_path: str = r"D:\Gen_ai\Car\Data") -> list[Document]:
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: '{folder_path}'")

        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found inside '{folder_path}'")

        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file, on_bad_lines="skip", engine="python")
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

    def create_vectorstore(self, documents: list[Document], force_rebuild: bool = False):
        if force_rebuild and os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)

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

    def query(self, question: str, k: int = 4):
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore not initialized. Call create_vectorstore() first.")
        return self.vectorstore.similarity_search(question, k=k)

    def generate_answer(self, question: str, chat_history: list[dict] = None, k: int = 4) -> dict:
        """Retrieve relevant cars, then ask the LLM to generate an answer, aware of prior turns."""
        results = self.query(question, k=k)
        context = "\n\n".join([doc.page_content for doc in results])

        # Format last few turns of history as plain text for the prompt
        history_text = "None yet — this is the first question."
        if chat_history:
            recent = chat_history[-6:]  # last 3 exchanges (user+assistant pairs)
            history_text = "\n".join(
                f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
                for turn in recent
            )

        chain = self.prompt | self.llm
        response = chain.invoke({
            "chat_history": history_text,
            "context": context,
            "question": question
        })

        return {
            "answer": response.content,
            "sources": [doc.page_content for doc in results]
        }