import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


class LangChainLLMInterface:
    def __init__(
        self,
        model_type="huggingface",
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        api_key=None,
    ):
        if model_type == "huggingface":
            llm_endpoint = HuggingFaceEndpoint(
                repo_id=model_name,
                task="text-generation",
                huggingfacehub_api_token=api_key or hf_token,
                temperature=0.3,
                max_new_tokens=250,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.2,
            )
            
            self.llm = ChatHuggingFace(llm=llm_endpoint)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def create_qa_chain(self, retriever):
        prompt_template = """
        <|system|>
        You are an HR Assistant named HRHelper. You provide helpful and conversational answers about company policies based on the information provided to you.
        </|system|>
        
        <|user|>
        I need information based on the following context:
        
        {context}
        
        My question is: {question}
        </|user|>
        
        <|assistant|>
        I'd be happy to help with your question about company HR policies!
        
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever.vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

        return qa_chain

    def generate_answer(self, qa_chain, query):
        result = qa_chain({"query": query})

        if result["result"] and isinstance(result["result"], str):
            lines = result["result"].split("\n")
            filtered_lines = [
                line for line in lines if not all(c == "-" for c in line.strip())
            ]
            result["result"] = "\n".join(filtered_lines)

            if not result["result"].strip():
                result["result"] = (
                    "Based on the company HR policies, I can provide you with the following information: "
                )
                if (result.get("source_documents") and len(result["source_documents"]) > 0):
                    for doc in result["source_documents"]:
                        if "answer" in doc.page_content.lower():
                            parts = doc.page_content.split("Answer:")
                            if len(parts) > 1:
                                result["result"] += parts[1].strip()

        return result
