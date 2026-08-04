import streamlit as st
from rag_pipeline import CarRAG

st.set_page_config(page_title="Indian Car Consultant", page_icon="🚗", layout="wide")

st.title("🚗 Indian Car Consultant Chatbot")
st.write("Ask me anything about cars — budget, mileage, features, comparisons. I'll remember our conversation.")


@st.cache_resource
def load_rag_system():
    rag = CarRAG()
    documents = rag.load_data(r"D:\Gen_ai\Car\Data")
    rag.create_vectorstore(documents)
    return rag


# Load the system once
with st.spinner("Loading car data and building RAG system... Please wait..."):
    try:
        rag = load_rag_system()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: clear chat button
with st.sidebar:
    st.header("Options")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("🔍 See retrieved car data used"):
                for i, source in enumerate(msg["sources"], 1):
                    st.write(f"**Option {i}**")
                    st.write(source)
                    st.divider()

# Chat input
question = st.chat_input("Example: Best car under 10 lakhs with good mileage")

if question:
    # Show user's message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Generate and show assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag.generate_answer(
                    question,
                    chat_history=st.session_state.messages[:-1],  # exclude the current question
                    k=4
                )
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
                st.stop()

        st.write(result["answer"])
        with st.expander("🔍 See retrieved car data used"):
            for i, source in enumerate(result["sources"], 1):
                st.write(f"**Option {i}**")
                st.write(source)
                st.divider()

    # Save assistant's response (with sources) to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })