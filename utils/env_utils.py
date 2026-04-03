import os

from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

MILVUS_URI = 'http://175.27.169.203:19530'
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')

COLLECTION_NAME = 't_collection02'
