import multiprocessing
import os
from multiprocessing import Queue

from documents.markdown_parser import MarkdownParser
from documents.milvus_db import MilvusVectorSave
from utils.log_utils import log


def file_parser_process(dir_path: str, output_queue: Queue, batch_size: int = 20):
    """进程1：解析目录下所有的md文件并放入队列中"""
    log.info(f"解析进程开始扫描目录:{dir_path}")

    """获取目录下所有的.md文件"""
    md_files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith('.md')
    ]

    if not md_files:
        log.warning("警告：未找到任何.md文件")
        output_queue.put(None)  # 发送终止信号
        return

    parser = MarkdownParser()

    doc_batch = []
    for md_file in md_files:
        try:
            docs = parser.parse_markdown_to_documents(md_file)
            print(f"{md_file}解析成功")
            if docs:
                doc_batch.extend(docs)
            if len(doc_batch) >= batch_size:
                output_queue.put(doc_batch)
                doc_batch.clear()  # 清空当前缓冲区所有的批次数据
        except Exception as e:
            log.error(f"解析失败{md_file}:{str(e)}")
            log.exception(e)

    # 发送剩余文档
    if doc_batch:
        output_queue.put(doc_batch)

    # 发送终止信号
    output_queue.put(None)
    log.info(f"解析完成，共处理完成了{len(md_files)}个文件")


def milvus_writer_process(input_queue: Queue):
    """进程2：从队列中读取并写入Milvus"""
    log.info("Milvus写入进程启动中...")

    mv = MilvusVectorSave()
    # mv.create_collection()
    total_count = 0
    while True:
        try:
            datas = input_queue.get()
            if datas is None:
                break

            if isinstance(datas,list):
                mv.insert_documents(datas)
                total_count += len(datas)
                log.info(f"目前写入了{total_count}条数据")
        except Exception as e:
            log.error(f"数据写入失败！")
            log.exception(e)

    log.info(f"总计写入了{total_count}条数据")

if __name__ == '__main__':
    # 配置参数
    md_dir = r'C:\Users\1\Desktop\RAG_PROJECT\datas\md'  # Markdown文件目录
    queue_maxsize = 20  # 队列最大容量（防止内存溢出）

    mv = MilvusVectorSave()
    mv.create_collection()

    # 创建进程间通信队列
    docs_queue = Queue(maxsize=queue_maxsize)

    # 启动子进程
    parser_proc = multiprocessing.Process(
        target=file_parser_process,
        args=(md_dir, docs_queue)
    )
    writer_proc = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue,)
    )

    parser_proc.start()
    writer_proc.start()

    # 等待进程结束
    parser_proc.join()
    writer_proc.join()

    print("系统提示：所有任务完成")
