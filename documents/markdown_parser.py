from typing import List
from langchain_experimental.text_splitter import SemanticChunker

from llm_models.embeddings_model import openai_embedding, bge_embedding
from utils.log_utils import log
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document


class MarkdownParser:
    """
    专门负责markdown文件的解析和切片
    """
    def __init__(self):
        self.text_splitter = SemanticChunker(
            bge_embedding, breakpoint_threshold_type="percentile"
        )

    def text_chunker(self, datas: List[Document]) -> List[Document]:
        new_docs = []
        for d in datas:
            if len(d.page_content) > 300:  # 内容超出了阈值，则按照语义再切割
                new_docs.extend(self.text_splitter.split_documents([d]))
                continue
            new_docs.append(d)
        return new_docs


    def parse_markdown_to_documents(self, md_file: str, encoding='utf-8') -> List[Document]:
        documents = self.parse_markdown(md_file)
        log.info(f'文件解析后的docs长度: {len(documents)}')

        merged_documents = self.merge_title_content(documents)

        log.info(f'文件合并后的长度: {len(merged_documents)}')

        chunk_documents = self.text_chunker(merged_documents)
        log.info(f'语义切割后的长度: {len(chunk_documents)}')
        return chunk_documents

    def parse_markdown(self, md_file: str) -> List[Document]:
        loader = UnstructuredMarkdownLoader(
            file_path=md_file,
            mode='elements',
            strategy='fast'
        )
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)

        return docs

    def merge_title_content(self, datas: List[Document]) -> List[Document]:
        merged_data = [] # 最终返回的整体的markdown内容
        parent_dict = {}  # 是一个字典，保存所有的父document， key为当前父document的ID
        # 清理document中的languages内容
        for document in datas:
            metadata = document.metadata
            if 'languages' in metadata:
                metadata.pop('languages')

            # 获取matadata中的三个关键元素parent_id、category、element_id
            parent_id = metadata.get('parent_id', None)
            category = metadata.get('category', None)
            element_id = metadata.get('element_id', None)

            # 判断类型是否为：内容document 且没有父document
            if category == 'NarrativeText' and parent_id is None:
                # 是内容就直接加入merged_data
                merged_data.append(document)
            # 判断类型是否为 标题
            if category == 'Title':
                # 是标题，直接将新建一个title 将内容存入metadata
                document.metadata['title'] = document.page_content
                # 判断是否有父document
                if parent_id in parent_dict:
                    #有则直接更新document，将父子document内容合并
                    document.page_content = parent_dict[parent_id].page_content + ' -> ' + document.page_content
                    # 将新的document内容存入parent_dict字典内
                    # 字典内包含：parent_id对应的document的父内容
                    #           element_id对应的新的document
                    #           matadata的title属性：原先自己的内容
                    #           page_content属性：父子标题拼接后最终的内容
                parent_dict[element_id] = document
            # 判断类型不是标题且有parent_id
            if category != 'Title' and parent_id:
                # 直接将内容与父内容直接拼接（也就是这个内容对应的上级标题）
                parent_dict[parent_id].page_content = parent_dict[parent_id].page_content + ' ' + document.page_content
                # 将父字典里的类型设为content
                parent_dict[parent_id].metadata['category'] = 'content'

        # 处理字典
        if parent_dict is not None:
            merged_data.extend(parent_dict.values())

        return merged_data


if __name__ == '__main__':
    file_path = r'C:\Users\1\Desktop\RAG_PROJECT\datas\md\tech_report_0tfhhamx.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)
    for item in docs:
        print(f"parentID: {item.metadata.get('parent_id',None)}")
        print(f"元数据: {item.metadata}")
        print(f"标题: {item.metadata.get('title', None)}")
        print(f"doc的内容: {item.page_content}\n")
        print("------" * 10)

# if __name__ == '__main__':
#     file_path = r'C:\Users\1\Desktop\RAG_PROJECT\datas\md\performance_faq.md'
#     parser = MarkdownParser()
#     docs = parser.parse_markdown(file_path)
#     print(f"合并前的doc数：{len(docs)}")
#     docs = parser.merge_title_content(docs)
#     print(f"合并后的doc数：{len(docs)}")
#     for item in docs:
#         print(f"元数据: {item.metadata}")
#         print(f"标题: {item.metadata.get('title', None)}")
#         print(f"doc的内容: {item.page_content}\n")
#         print("------" * 10)
