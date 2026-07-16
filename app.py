import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# 1. Page Configuration
st.set_page_config(page_title="Academic RAG Bot", page_icon="🎓", layout="centered")

# 2. CSS Styling (This brings back your dark theme/styling)
st.markdown("""
    <style>
    .main { background-color: transparent; }
    .stApp { background-color: #0e1117; }
    [data-testid="stChatMessage"] { 
        background-color: #262730; 
        border-radius: 15px; 
        padding: 15px; 
        margin-bottom: 10px;
    }
    h1 { color: #ffffff; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- Configuration ---
COLLECTION_NAME = "academic_docs"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b-instruct-q4_k_m"

# --- Helper Functions ---
@st.cache_resource
def get_retriever():
    client = QdrantClient(path="./local_qdrant")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://localhost:11434")
    return QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings).as_retriever(search_kwargs={"k": 3})

# --- Sidebar (This brings back the Sidebar) ---
with st.sidebar:
    st.header("🎓 Academic Bot")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Interaction ---
st.title("Academic RAG Assistant 🤖")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History (This brings back the visual chat bubbles)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for src in message["sources"]: st.write(f"- {src}")

# Chat Input
if prompt := st.chat_input("Ask a question about your study material..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                retriever = get_retriever()
                docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in docs])
                sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))

                history_text = ""
                for msg in st.session_state.messages[-5:-1]:
                    history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

                llm = ChatOllama(model=LLM_MODEL, base_url="http://localhost:11434")
                full_prompt = f"Conversation History:\n{history_text}\nContext:\n{context}\nQuestion: {prompt}"
                
                response = llm.invoke(full_prompt)
                
                st.markdown(response.content)
                with st.expander("📚 View Sources"):
                    for src in sources: st.write(f"- {src}")

                st.session_state.messages.append({"role": "assistant", "content": response.content, "sources": sources})
            except Exception as e:
                st.error(f"Error: {e}")
