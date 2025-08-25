# RAG-Based HR Policy FAQ System

A Retrieval-Augmented Generation (RAG) based Question Answering system designed to answer HR policy questions. This system uses natural language processing techniques to retrieve relevant information from a knowledge base and generate accurate answers to HR policy questions.

## Project Overview

This system combines the power of large language models with vector search to provide accurate answers to questions about HR policies. It uses:

- Document preprocessing and chunking
- Embedding-based vector storage
- Semantic retrieval
- LLM-based answer generation
- Streamlit web interface

## Installation

### Prerequisites

- Python 3.10+
- pip package manager

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd rag-based-qa-system
```

### Step 2: Create and activate a virtual environment

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use: myenv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure HuggingFace Access Token

Create a `.env` file in the root directory and add the following variable:

```
HUGGINGFACEHUB_ACCESS_TOKEN=your_access_token_here
```

Note: For restricted(gated) models you have to request access on HuggingFace

Make sure to replace `your_access_token_here` with your actual Hugging Face Hub access token.

The application will automatically load this token from the `.env` file when required.

## Running the Application

1. Make sure your virtual environment is activated
2. Run the Streamlit application:

```bash
streamlit run src/main.py
```

3. Open your web browser and go to http://localhost:8501

## System Architecture

### Components

1. **Data Processing (`src/data/`):**
   - Handles loading and preprocessing of documents
   - Splits documents into manageable chunks
   - Cleans text data for better processing

2. **Vector Storage (`src/retrieval/vectore_store.py`):**
   - Converts document chunks into vector embeddings
   - Stores embeddings for efficient retrieval
   - Uses sentence-transformers for embedding generation

3. **Retriever (`src/retrieval/retriever.py`):**
   - Performs semantic search to find relevant documents
   - Retrieves top-k most similar documents to the query

4. **LLM Interface (`src/generation/llm_interface.py`):**
   - Interfaces with Hugging Face models
   - Generates natural language answers based on retrieved context

5. **Web Interface (`src/main.py`):**
   - Streamlit-based user interface
   - Handles user queries and displays responses
   - Shows source documents for transparency

6. **Evaluation System (`src/evaluation/`):**
   - Evaluates the system's performance with ground truth data
   - Tracks correct answers and "I don't know" responses
   - Reports accuracy metrics and detailed analysis

### Data Flow

1. User inputs a question via the Streamlit interface
2. The question is processed and embedded
3. The retriever component finds relevant documents from the vector store
4. These documents are passed to the LLM along with the original question
5. The LLM generates a response based on the retrieved context
6. The response is displayed to the user along with source references


## TODOs

1. **Model Optimization:**
   - Implement quantization for better performance on CPU
   - Add support for more efficient embedding models

2. **Enhanced Retrieval:**
   - Implement hybrid search (keyword + semantic)
   - Add re-ranking of retrieved documents

3. **User Experience:**
   - Implement chat history persistence
   - Add user feedback mechanism for answer quality

4. **Evaluation:**
   - Create evaluation pipeline with metrics (ROUGE, BLEU)
   - Implement A/B testing for different retrieval strategies

5. **Data Management:**
   - Add support for dynamic document updates
   - Implement document source tracking
