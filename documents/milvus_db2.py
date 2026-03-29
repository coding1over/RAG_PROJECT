from typing import List
import logging

from langchain_core.documents import Document
from langchain_milvus import Milvus
# 只保留必要依赖，彻底避免连接冲突

# 你的自定义模块
from documents.markdown_parser import MarkdownParser
from llm_models.embeddings_model import bge_embedding
from utils.env_utils import MILVUS_URI, COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusVectorSave:
    def __init__(self) -> None:
        self.vector_store = None

    # 🔥 核心：让LangChain自动创建连接+集合+索引，零冲突
    def init_milvus(self):
        self.vector_store = Milvus(
            embedding_function=bge_embedding,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_URI},
            # 自动创建字段+向量索引，完全适配Milvus 2.4.8
            auto_id=True,
            # 关键：本地模式，不搞复杂ORM，彻底解决连接错误
            primary_field="id",
            text_field="text",
            vector_field="dense",
        )
        logger.info("✅ Milvus 初始化完成（自动建表+连接）")

    def add_documents(self, datas: List[Document]):
        self.vector_store.add_documents(datas)
        logger.info(f"✅ 成功插入 {len(datas)} 条文档！")

if __name__ == '__main__':
    # 解析文档
    file_path = r'C:\Users\1\Desktop\RAG_PROJECT\datas\md\tech_report_0tfhhamx.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    # 初始化并插入数据（全程无手动连接，无冲突）
    mv = MilvusVectorSave()
    mv.init_milvus()
    mv.add_documents(docs)

    logger.info("🎉 RAG向量库搭建完成！")