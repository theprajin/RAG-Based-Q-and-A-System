import os
import json
import time
import streamlit as st

from data.preprocessing import preprocess_dataset
from retrieval.vectore_store import LangChainVectorStore
from retrieval.retriever import LangChainRetriever
from generation.llm_interface import LangChainLLMInterface
from config import API_KEY, MODEL_NAME, EMBEDDING_MODEL, MAX_RETRIEVAL_RESULTS

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["FORCE_CPU"] = "1"

st.set_page_config(
    page_title="HR Policy FAQ Assistant",
    page_icon="👔",
    layout="centered",
)

if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm_interface" not in st.session_state:
    st.session_state.llm_interface = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

st.title("HR Policy FAQ Assistant")
st.markdown(
    """
Ask questions about company HR policies and get accurate answers based on our knowledge base.
"""
)


def initialize_rag_system():
    with st.spinner("Setting up the HR assistant..."):
        try:
            dataset_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data",
                "raw",
                "sample_dataset.json",
            )

            if not os.path.exists(dataset_path):
                st.error(
                    "HR policy dataset not found. Please make sure the sample_dataset.json file exists."
                )
                return False

            with open(dataset_path, "r") as f:
                faq_data = json.load(f)

            temp_dataset_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data",
                "processed",
                "temp_hr_faqs.json",
            )
            os.makedirs(os.path.dirname(temp_dataset_path), exist_ok=True)

            docs = []
            for item in faq_data.get("faqs", []):
                doc = {
                    "title": item.get("question", ""),
                    "content": f"Question: {item.get('question', '')}\nAnswer: {item.get('answer', '')}",
                }
                docs.append(doc)

            with open(temp_dataset_path, "w") as f:
                json.dump(docs, f)

            processed_docs = preprocess_dataset(
                temp_dataset_path, is_json=True, chunk_size=1000, chunk_overlap=200
            )

            vector_store = LangChainVectorStore(embedding_model_name=EMBEDDING_MODEL)
            vector_store.create_from_documents(processed_docs)

            retriever = LangChainRetriever(vector_store)

            api_key_value = API_KEY if API_KEY != "your_api_key_here" else None
            llm_interface = LangChainLLMInterface(
                model_type="huggingface",
                model_name=(
                    MODEL_NAME
                    if MODEL_NAME != "huggingface_model_name"
                    else "mistralai/Mistral-7B-Instruct-v0.3"
                ),
                api_key=api_key_value,
            )

            qa_chain = llm_interface.create_qa_chain(vector_store)

            st.session_state.vector_store = vector_store
            st.session_state.retriever = retriever
            st.session_state.llm_interface = llm_interface
            st.session_state.qa_chain = qa_chain
            st.session_state.initialized = True

            return True

        except Exception as e:
            st.error(f"Error initializing the HR assistant: {str(e)}")
            return False


if not st.session_state.initialized:
    if initialize_rag_system():
        st.success("HR assistant is ready to answer your questions!")
    else:
        st.error(
            "Failed to initialize the HR assistant. Please check the errors above."
        )
        st.stop()

st.subheader("Ask about HR policies:")
user_question = st.text_area(
    "Type your question here:",
    placeholder="E.g., How do I request vacation days?",
    height=100,
)

if st.button("Get Answer", type="primary"):
    if not user_question:
        st.warning("Please enter a question about HR policies.")
    else:
        with st.spinner("Searching for an answer..."):
            try:
                start_time = time.time()

                relevant_docs = st.session_state.retriever.retrieve(
                    user_question, top_k=MAX_RETRIEVAL_RESULTS
                )

                result = st.session_state.llm_interface.generate_answer(
                    st.session_state.qa_chain, user_question
                )

                answer = result["result"]
                source_docs = result["source_documents"]

                end_time = time.time()
                response_time = round(end_time - start_time, 2)

                st.markdown("### Answer:")
                st.markdown(
                    f"""<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; color: #333333; font-size: 16px;">
                               {answer}
                               </div>""",
                    unsafe_allow_html=True,
                )

                with st.expander("View sources"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(f"```\n{doc.page_content}\n```")

                st.caption(f"Response generated in {response_time} seconds")

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

st.markdown("---")
st.markdown("HR Policy FAQ Assistant | Powered by RAG technology")
