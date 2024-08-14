# generate.py：这个文件会在目录中生成 PDF 文档的嵌入，并将它们保存在 Milvus 集合中。
import logging, os
import pymilvus
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceWindowNodeParser
import re


# load_dotenv("E:\\data\\rag-microservices\\rag-microservices\\ingestion-service\\.env.example")
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../ingestion-service/.env.example'))

# 读取环境配置文件
load_dotenv(dotenv_path=env_path)
# load_dotenv("../../../ingestion-service/.env.example")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

model = os.getenv("MODEL", "gpt-3.5-turbo")
llm=OpenAI(model=model)
embed_model=OpenAIEmbedding(model="text-embedding-ada-002")

def sentence_splitter(text):
    nodes = re.split("(?<=。)|(?<=？)|(?<=！)", text)
    nodes = [node for node in nodes if node]
    return nodes

def generate_datasource():

    try:
        milvus_uri = os.getenv("MILVUS_URI")
        milvus_api_key = os.getenv("MILVUS_API_KEY")
        milvus_collection = os.getenv("MILVUS_COLLECTION")
        milvus_dimension = int(os.getenv("MILVUS_DIMENSION"))

        if not all([milvus_uri, milvus_api_key, milvus_collection, milvus_dimension]):
            raise ValueError("Missing required environment variables.")

        # 创建 MilvusVectorStore 
        vector_store = MilvusVectorStore(
            uri=milvus_uri,
            token=milvus_api_key,
            collection_name=milvus_collection,
            dim=milvus_dimension, # 如果本身没有创建，则强制创建
            overwrite=True,  
        )

        # 创建 StorageContext 
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 创建解析器 使用句子窗口检索 以文档句子为单位进行切分
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
            sentence_splitter=sentence_splitter
        )

        documents = SimpleDirectoryReader(input_files=["E:\\data\\rag-microservices-agent\\rag-microservices\\ingestion-service\\skin_care_products.pdf"]).load_data()
        nodes = node_parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid environment variables: {e}")
    except ConnectionError as e:
        raise ConnectionError(f"Failed to connect to Milvus: {e}")


if __name__ == "__main__":
    generate_datasource()
