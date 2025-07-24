from langchain_groq import ChatGroq
from app.config.config import GROQ_API_KEY, HF_TOKEN, HUGGINGFACE_REPO_ID
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os




load_dotenv()  # loads from .env
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

logger = get_logger(__name__)

from huggingface_hub import InferenceClient
from langchain.llms.base import LLM
from typing import Optional, List
from app.common.logger import get_logger
from app.common.custom_exception import CustomException



class HFConversationalLLM(LLM):
    model_name: str
    hf_token: str
    client: InferenceClient

    def __init__(self, model_name: str, hf_token: str):
        self.model_name = model_name
        self.hf_token = hf_token
        self.client = InferenceClient(
            model=model_name,
            token=hf_token
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        logger.info("Calling HF conversational model directly...")
        response = self.client.conversational(
            inputs=prompt,
            parameters={"temperature": 0.3, "max_new_tokens": 256}
        )
        return response.generated_responses[-1] if response.generated_responses else ""

    @property
    def _llm_type(self) -> str:
        return "huggingface-conversational"



def hf_load_llm(hf_repo_id: str = "HuggingFaceH4/zephyr-7b-alpha", hf_token: str = os.getenv("HF_ACCESS_TOKEN")):
    try:
        logger.info(f"Loading LLM from Hugging Face: {hf_repo_id} with conversational task")
        if not hf_token:
            raise ValueError("HF_ACCESS_TOKEN is not set")

        llm = HuggingFaceEndpoint(
            repo_id=hf_repo_id,
            task="conversational",  # ✅ required for Zephyr
            huggingfacehub_api_token=hf_token,
            temperature=0.3,
            max_new_tokens=256,
            return_full_text=False
        )

        logger.info("LLM loaded successfully from Hugging Face.")
        return llm

    except Exception as e:
        logger.error(f"Failed to load an LLM from Hugging Face | Error: {e}")
        return None


# def hf_load_llm(hf_repo_id: str= HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN ):
#     try: 
#         logger.info("Load LLM from HuggingFace")
#         llm = HuggingFaceEndpoint(
#             repo_id = "HuggingFaceH4/zephyr-7b-alpha",
#             temperature= 0.3, # randomness
#             max_new_tokens = 256, # expected response lenth
#             return_full_text = False,
#             huggingfacehub_api_token = HF_TOKEN,
#             task="conversational" 
#         )
#         logger.info("LLM loaded successfully from Hugging Face.")

#         return llm

#     except Exception as e:
#         error_message = CustomException("Failed to load an LLM from HuggingFace", e)
#         logger.error(str(error_message))
#         return None
    



def load_llm(model_name: str = "llama-3.1-8b-instant", groq_api_key: str = GROQ_API_KEY):
    try:
        logger.info("Loading LLM from Groq using LLaMA3 model...")

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.3,
            max_tokens=256,
        )

        logger.info("LLM loaded successfully from Groq.")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load an LLM from Groq", e)
        logger.error(str(error_message))
        return None
    
    
    
# from langchain_huggingface import ChatHuggingFace
# from langchain_huggingface.llms import HuggingFaceEndpoint
# import os

# HF_TOKEN = os.getenv("HF_ACCESS_TOKEN")

# llm = ChatHuggingFace(
#     llm=HuggingFaceEndpoint(
#         repo_id="HuggingFaceH4/zephyr-7b-alpha",
#         huggingfacehub_api_token=HF_TOKEN,
#         task="conversational"
#     ),
#     temperature=0.3,
#     max_new_tokens=256
# )

# response = llm.invoke([
#     {"role": "user", "content": "What are the symptoms of malaria?"}
# ])

# print("✅ Response:", response)
