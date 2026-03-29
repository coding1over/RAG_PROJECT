from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from utils.env_utils import OPENAI_API_KEY, OPENAI_BASE_URL


# ========== 云端Embedding（需API Key+VPN，仅测试/备用） ==========
openai_embedding = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
    model="text-embedding-3-small"
)


# ========== 本地Embedding（主力使用，无API Key/VPN依赖） ==========

model_name = "BAAI/bge-small-zh-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
bge_embedding = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs # 关键参数：指定缓存路径
)