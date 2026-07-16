Academic RAG Assistant

A specialized Retrieval-Augmented Generation (RAG) assistant designed to help university students interact with their academic documents.
This tool bridges the gap between static PDF notes and an intelligent, conversational interface, enabling efficient study sessions and quick information retrieval.

🚀 Key Features

Intelligent Retrieval: Utilizes RAG architecture to fetch precise context from academic PDFs before generating answers.

Context-Aware: Reduces hallucinations by forcing the model to answer based strictly on the provided syllabus and course materials.

Efficient Local Database: Implemented using Qdrant for high-performance vector storage and retrieval.

Streamlined Pipeline: Features a custom ingest.py script for automated document chunking and embedding generation.
 




🛠 Tech Stack

Language: Python

Frameworks: LangChain, Streamlit

Vector Database: Qdrant

LLM Integration: Ollama

🏗 Project Architecture

Ingestion: ingest.py processes raw PDFs, chunks text into manageable segments, and generates vector embeddings.

Storage: Embeddings are indexed and stored in a local Qdrant vector database for fast similarity search.

Retrieval & Generation: app.py captures user queries, performs a semantic search in the vector store, and provides the LLM with relevant document context for accurate responses.

💻 How to Run

Clone the repository:

    git clone https://github.com/Abhinav-Dhananjaya/academic-rag-assistant.git

    cd academic-rag-assistant

Install dependencies:

    pip install -r requirements.txt

Ingest your documents:

    Place your PDFs in the data/ folder and run:

    python ingest.py

Launch the app:

    streamlit run app.py

🎓 Academic Context

This project was developed to optimize study workflows for engineering students, specifically tackling the challenge of parsing dense technical documentation. It demonstrates practical skills in Retrieval-Augmented Generation (RAG), Vector Database management, and full-stack AI application deployment.
