# from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from utils.env_utils import OPENAI_API_KEY, OPENAI_BASE_URL
#
#
# # ========== 云端Embedding（需API Key+VPN，仅测试/备用） ==========
# openai_embedding = OpenAIEmbeddings(
#     openai_api_key=OPENAI_API_KEY,
#     openai_api_base=OPENAI_BASE_URL,
#     model="text-embedding-3-small"
# )
#
#
# # ========== 本地Embedding（主力使用，无API Key/VPN依赖） ==========
#
# model_name = "BAAI/bge-small-zh-v1.5"
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings": True}
# bge_embedding = HuggingFaceEmbeddings(
#     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs # 关键参数：指定缓存路径
# )

import os
import torch
from typing import List
from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings
# ✅ 无红行的导入（适配低版本LangChain）
from pydantic import BaseModel, Extra
from langchain_openai import OpenAIEmbeddings
from utils.env_utils import OPENAI_API_KEY, OPENAI_BASE_URL

# ========== 1. 强制离线模式（禁用所有HuggingFace网络请求） ==========
os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ========== 2. 封装原生BGE模型为LangChain兼容的Embedding类 ==========
class BGEEmbeddings(BaseModel, Embeddings):
    """
    基于Transformers原生接口封装BGE模型，兼容LangChain Embeddings协议
    彻底避开sentence-transformers的格式依赖（CPU环境专用）
    """
    model_path: str  # 本地BGE模型路径
    device: str = "cpu"
    normalize_embeddings: bool = True
    tokenizer: AutoTokenizer = None
    model: AutoModel = None

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ✅ 修复：移除device_map，CPU环境直接加载
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,  # 强制只读本地文件
            trust_remote_code=True
        )
        # ✅ 核心修复：删除device_map参数，CPU环境无需
        self.model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True
        ).to(self.device)  # CPU环境直接指定设备

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """核心：生成BGE模型的Embedding（遵循官方标准用法）"""
        # BGE模型要求：文本前加固定前缀，提升检索效果
        texts = [f"为这个句子生成表示以用于检索：{text}" for text in texts]
        # 编码文本
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        # 推理生成向量
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 取cls向量（BGE模型的标准做法）
        embeddings = outputs.last_hidden_state[:, 0]
        # 归一化（可选，提升检索效果）
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # 转换为列表返回
        return embeddings.cpu().numpy().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """实现LangChain Embeddings接口：批量嵌入文档"""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """实现LangChain Embeddings接口：嵌入单条查询"""
        return self._embed([text])[0]


# ========== 3. 初始化云端OpenAI Embedding（备用） ==========
openai_embedding = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
    model="text-embedding-3-small"
)

# ========== 4. 初始化本地BGE Embedding（主力，彻底离线） ==========
# 替换为你镜像下载的本地模型路径
LOCAL_BGE_PATH = r"D:\my_project\RAG_PROJECT\models\huggingface_cache\models--BAAI--bge-small-zh-v1.5"
bge_embedding = BGEEmbeddings(
    model_path=LOCAL_BGE_PATH,
    device="cpu",  # 明确指定CPU，无需accelerate
    normalize_embeddings=True
)
