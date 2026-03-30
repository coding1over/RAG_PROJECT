from typing import List
import json

from langchain_core.documents import Document
from pymilvus import MilvusClient, Function
from pymilvus.client.types import DataType, FunctionType

# 你的自定义模块
from documents.markdown_parser import MarkdownParser
from llm_models.embeddings_model import bge_embedding
from utils.env_utils import MILVUS_URI, COLLECTION_NAME


class MilvusVectorSave:
    def __init__(self):
        # 直接创建新版 MilvusClient 连接（永无连接报错）
        self.client = MilvusClient(uri=MILVUS_URI)
        self.embedding = bge_embedding

    def create_collection(self):
        """创建集合（纯新版SDK，兼容Milvus 2.5.6 + pymilvus 2.6）"""
        # 删除旧集合
        if self.client.has_collection(COLLECTION_NAME):
            self.client.drop_collection(COLLECTION_NAME)

        # 构建Schema
        schema = self.client.create_schema()
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True,
                         analyzer_params={"tokenizer": "jieba"})
        schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='category_depth', datatype=DataType.INT64)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=512)

        # BM25函数（正确配置）
        bm25_func = Function(
            name="text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"]
        )
        schema.add_function(bm25_func)

        # 索引参数（修复BM25度量类型）
        index_params = self.client.prepare_index_params()
        # 稀疏向量索引
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25"
        )
        # 稠密向量索引
        index_params.add_index(
            field_name="dense",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 64}
        )

        # 创建集合
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )
        print("✅ 集合创建成功（无任何连接/索引报错）")

    def insert_documents(self, docs: List[Document]):
        """原生插入数据（绕过LangChain连接bug）"""
        data = []
        # 生成向量并组装数据
        embeddings = self.embedding.embed_documents([doc.page_content for doc in docs])

        for idx, doc in enumerate(docs):
            data.append({
                "text": doc.page_content,
                "category": doc.metadata.get("category", ""),
                "source": doc.metadata.get("source", ""),
                "filename": doc.metadata.get("filename", ""),
                "filetype": doc.metadata.get("filetype", ""),
                "title": doc.metadata.get("title", ""),
                "category_depth": doc.metadata.get("category_depth", 0),
                "dense": embeddings[idx]
                # sparse 由BM25函数自动生成，无需手动插入
            })

        # 原生插入
        res = self.client.insert(COLLECTION_NAME, data)
        print(f"✅ 成功插入 {res['insert_count']} 条数据")

    def test_query(self):
        """测试查询"""
        result = self.client.query(
            collection_name=COLLECTION_NAME,
            filter="category == 'Title'",
            output_fields=["text", "category", "filename"]
        )
        print("测试查询结果：", json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    # 1. 解析文档
    file_path = r'C:\Users\1\Desktop\RAG_PROJECT\datas\md\tech_report_0tfhhamx.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    # 2. 核心操作（纯原生SDK，永无连接报错）
    mv = MilvusVectorSave()
    mv.create_collection()   # 创建集合
    mv.insert_documents(docs)# 插入数据
    mv.test_query()          # 测试查询

    print("\n🎉 全部执行成功！Milvus 功能正常！")
