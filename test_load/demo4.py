from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader(
    file_path=r'C:\Users\1\Desktop\RAG_PROJECT\datas\md\operational_faq.md',
    mode='elements',
    strategy='fast'
)

docs = loader.load()
print(f'doc的数量是: {len(docs)}')

for i in range(10):

    print(docs[i].metadata)
    print(docs[i].page_content)


    # print(docs[i].metadata.get('parent_id','无parentId'))
    # print(docs[i].metadata.get('title','无parentId'))

    print('--' * 50)