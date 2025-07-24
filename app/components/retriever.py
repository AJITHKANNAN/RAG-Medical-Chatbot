from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint

from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import HF_TOKEN

logger = get_logger(__name__)

def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()
        logger.info("Vectorstore successfully converted to retriever")
        if db is None:
            raise CustomException("Vector store not found or failed to load")

        retriever = db.as_retriever(search_kwargs={"k": 5})

        logger.info("Loading LLM from Hugging Face")
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="HuggingFaceH4/zephyr-7b-alpha",
                huggingfacehub_api_token=HF_TOKEN,
                task="conversational"
            ),
            temperature=0.3,
            max_new_tokens=256
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        logger.info("QA chain created successfully")
        
        return qa_chain

    except Exception as e:
        logger.error(f"Failed to create QA chain | Error: {str(e)}")
        return None
