from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import IndexType, MilvusClient, Function
from pymilvus.client.types import MetricType, DataType, FunctionType

from documents.markdown_parser import MarkdownParser
from llm_models.embeddings_model import bge_embedding
from utils.env_utils import MILVUS_URI, COLLECTION_NAME


class MilvusVectorSave:
    """把新的document数据插入到数据库中"""

    def __init__(self) -> object:
        """自定义collection的索引"""
        self.vector_store_saved: Milvus = None

    def create_collection(self):
        client = MilvusClient(uri=MILVUS_URI)
        schema = client.create_schema()
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True,
                         analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
        schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='category_depth', datatype=DataType.INT64)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=512)

        bm25_function = Function(
            name="text_bm25_emb",  # Function name
            input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,  # Set to `BM25`
        )
        schema.add_function(bm25_function)
        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
            metric_type=MetricType.IP,
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
                "bm25_k1": 1.2,
                "bm25_b": 0.75
            },
        )
        index_params.add_index(
            field_name="dense",
            index_name="dense_inverted_index",
            index_type=IndexType.HNSW,  # Inverted index type for sparse vectors
            metric_type="IP",
            params={"M": 16, "efConstruction": 64}  # M :邻接节点数, efConstruction: 搜索范围
        )

        if COLLECTION_NAME in client.list_collections():
            # 先释放， 再删除索引，再删除collection
            client.release_collection(collection_name=COLLECTION_NAME)
            client.drop_index(collection_name=COLLECTION_NAME, index_name='sparse_inverted_index')
            client.drop_index(collection_name=COLLECTION_NAME, index_name='dense_inverted_index')
            client.drop_collection(collection_name=COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def create_connection(self):
        """创建一个Connection： milvus + langchain。pip install  langchain-milvus"""
        self.vector_store_saved = Milvus(
            embedding_function=bge_embedding,
            collection_name=COLLECTION_NAME,
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense', 'sparse'],
            consistency_level="Strong",
            auto_id=True,
            connection_args={"uri": MILVUS_URI}
        )
        print("连接已建立")

    def add_documents(self, datas: List[Document]):
        """把新的document保存到Milvus中"""
        self.vector_store_saved.add_documents(datas)



if __name__ == '__main__':
    # 解析文件内容
    file_path = r'C:\Users\1\Desktop\RAG_PROJECT\datas\md\tech_report_0tfhhamx.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    # 写入Milvus数据库
    mv = MilvusVectorSave()
    mv.create_collection()
    mv.create_connection()
    mv.add_documents(docs)

    client = mv.vector_store_saved.client
    # 得到表结构
    desc_collection = client.describe_collection(
        collection_name=COLLECTION_NAME
    )
    print('表结构是: ', desc_collection)

    # 得到当前表的，所有的index
    res = client.list_indexes(
        collection_name=COLLECTION_NAME
    )
    print('表中的所有索引：', res)

    if res:
        for i in res:
            # 得到索引的描述
            desc_index = client.describe_index(
                collection_name=COLLECTION_NAME,
                index_name=i
            )
            print(desc_index)

    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="category == 'Title'",  # 查询 category == 'Title' 的所有数据
        output_fields=['text', 'category', 'filename']  # 指定返回的字段
    )

    print('测试 过滤查询的结果是: ', result)
# from typing import List
# import logging
#
# from langchain_core.documents import Document
# from langchain_milvus import Milvus, BM25BuiltInFunction
# from pymilvus import MilvusClient, Function
# from pymilvus.client.types import MetricType, DataType, FunctionType
#
# # 自定义模块（根据你的项目路径调整）
# from documents.markdown_parser import MarkdownParser
# from llm_models.embeddings_model import bge_embedding
# from utils.env_utils import MILVUS_URI, COLLECTION_NAME
#
# # 配置日志（方便排查问题）
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# class MilvusVectorSave:
#     """把新的document数据插入到Milvus数据库中"""
#
#     def __init__(self) -> None:
#         self.vector_store_saved: Milvus = None
#
#     def create_collection(self):
#         """创建集合 + 正确配置索引（核心修复）"""
#         client = MilvusClient(uri=MILVUS_URI)
#
#         # 1. 构建Schema（明确字段类型，避免解析歧义）
#         schema = client.create_schema(
#             auto_id=True,
#             primary_field_name="id"
#         )
#         # 主键字段（INT64）
#         schema.add_field(
#             field_name="id",
#             datatype=DataType.INT64,
#             is_primary=True,
#             auto_id=True
#         )
#         # 文本字段（VARCHAR，启用jieba分词）
#         schema.add_field(
#             field_name="text",
#             datatype=DataType.VARCHAR,
#             max_length=6000,
#             enable_analyzer=True,
#             analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]}
#         )
#         # 其他VARCHAR字段（无索引，避免冲突）
#         for field in ["category", "source", "filename", "filetype", "title"]:
#             schema.add_field(
#                 field_name=field,
#                 datatype=DataType.VARCHAR,
#                 max_length=1000
#             )
#         # 数值字段
#         schema.add_field(
#             field_name="category_depth",
#             datatype=DataType.INT64
#         )
#         # 稀疏向量字段（BM25生成）
#         schema.add_field(
#             field_name="sparse",
#             datatype=DataType.SPARSE_FLOAT_VECTOR
#         )
#         # 稠密向量字段（BGE嵌入，维度需与embedding模型一致！）
#         schema.add_field(
#             field_name="dense",
#             datatype=DataType.FLOAT_VECTOR,
#             dim=512  # 确认bge_embedding的输出维度是512，否则改这里
#         )
#
#         # 2. 配置BM25函数（text → sparse向量）
#         bm25_function = Function(
#             name="text_bm25_emb",
#             input_field_names=["text"],
#             output_field_names=["sparse"],
#             function_type=FunctionType.BM25,
#             # 显式指定BM25参数，避免与索引参数冲突
#             params={"bm25_k1": 1.2, "bm25_b": 0.75}
#         )
#         schema.add_function(bm25_function)
#
#         # 3. 构建索引参数（核心：字段与索引严格匹配）
#         index_params = client.prepare_index_params()
#
#         # ① 稀疏向量索引（仅作用于sparse字段）
#         index_params.add_index(
#             field_name="sparse",
#             index_name="sparse_idx",
#             index_type="SPARSE_INVERTED_INDEX",
#             metric_type=MetricType.IP,  # 必须用MetricType枚举，不能用字符串
#             params={
#                 "inverted_index_algo": "DAAT_MAXSCORE"
#                 # BM25参数移到Function里，这里只保留索引算法
#             }
#         )
#
#         # ② 稠密向量索引（仅作用于dense字段，统一用字符串类型）
#         index_params.add_index(
#             field_name="dense",
#             index_name="dense_hnsw_idx",
#             index_type="HNSW",  # 统一用字符串，避免枚举解析错误
#             metric_type=MetricType.IP,
#             params={"M": 16, "efConstruction": 64}
#         )
#
#         # 4. 清理旧集合（简化逻辑，drop_collection会自动删索引）
#         if COLLECTION_NAME in client.list_collections():
#             logger.info(f"删除旧集合：{COLLECTION_NAME}")
#             client.drop_collection(collection_name=COLLECTION_NAME)
#
#         # 5. 创建集合（关键：明确索引仅作用于向量字段）
#         client.create_collection(
#             collection_name=COLLECTION_NAME,
#             schema=schema,
#             index_params=index_params,
#             shards_num=1  # 单机版固定1分片
#         )
#         logger.info(f"集合 {COLLECTION_NAME} 创建成功")
#
#     def create_connection(self):
#         """创建LangChain-Milvus连接"""
#         self.vector_store_saved = Milvus(
#             embedding_function=bge_embedding,
#             collection_name=COLLECTION_NAME,
#             builtin_function=BM25BuiltInFunction(),
#             vector_field=["dense", "sparse"],  # 双向量检索
#             consistency_level="Strong",
#             auto_id=True,
#             connection_args={"uri": MILVUS_URI}
#         )
#         logger.info("Milvus连接已建立")
#
#     def add_documents(self, datas: List[Document]):
#         """插入文档到Milvus"""
#         if not self.vector_store_saved:
#             raise RuntimeError("请先调用create_connection建立连接")
#         logger.info(f"开始插入 {len(datas)} 条文档")
#         self.vector_store_saved.add_documents(datas)
#         logger.info("文档插入完成")
#
# if __name__ == '__main__':
#     # 1. 解析Markdown文件
#     file_path = r'C:\Users\1\Desktop\RAG_PROJECT\datas\md\tech_report_0tfhhamx.md'
#     parser = MarkdownParser()
#     docs = parser.parse_markdown_to_documents(file_path)
#     logger.info(f"解析得到 {len(docs)} 条文档")
#
#     # 2. 初始化并写入Milvus
#     mv = MilvusVectorSave()
#     mv.create_collection()
#     mv.create_connection()
#     mv.add_documents(docs)
#
#     # 3. 验证集合和索引
#     client = mv.vector_store_saved.client
#     # 打印集合结构
#     desc_collection = client.describe_collection(COLLECTION_NAME)
#     logger.info(f"集合结构：{desc_collection}")
#     # 打印所有索引
#     indexes = client.list_indexes(COLLECTION_NAME)
#     logger.info(f"集合索引列表：{indexes}")
#     for idx_name in indexes:
#         idx_desc = client.describe_index(COLLECTION_NAME, idx_name)
#         logger.info(f"索引 {idx_name} 详情：{idx_desc}")
#
#     # 4. 测试过滤查询
#     result = client.query(
#         collection_name=COLLECTION_NAME,
#         filter="category == 'Title'",
#         output_fields=["text", "category", "filename"]
#     )
#     logger.info(f"过滤查询结果（category=Title）：{result}")
