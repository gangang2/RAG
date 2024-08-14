from llama_index.core.tools import QueryEngineTool
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

from typing import Tuple


async def create_doc_tools(doc_tp: str, doc_name: str, verbose: bool = True) -> Tuple[QueryEngineTool, QueryEngineTool]:
    """
    为文档创建摘要索引工具 和 向量索引工具 以满足不同检索需求
    """
    try:
        document = SimpleDirectoryReader(input_files=[doc_tp]).load_data()
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(document)

        Settings.llm = OpenAI(model="gpt-3.5-turbo")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

        summary_index = SummaryIndex(nodes)
        vector_index = VectorStoreIndex(nodes)

        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        vector_query_engine = vector_index.as_query_engine()

        response = summary_query_engine.query("虚拟偶像的摘要")
        print(f"Response：${response.response}")

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            name=f"{doc_name}_summury_query_engine_tool",
            description=f"{doc_name} is a document that contains information about {doc_name}. Use this tool to answer summary questions about {doc_name}.",
        )

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            name=f"{doc_name}_vector_query_engine_tool",
            description=f"{doc_name} is a document that contains information about {doc_name}. Use this tool to answer context retrieval questions about {doc_name}.",
        )

    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid environment variables: {e}")
    except ConnectionError as e:
        raise ConnectionError(f"Failed to connect to OpenAI: {e}")

    return summary_tool, vector_tool

# import asyncio
#
# from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
# from llama_index.core.selectors import LLMSingleSelector
# import glob
# import os
# from pathlib import Path
# from llama_index.core.agent import FunctionCallingAgentWorker
# from llama_index.core.agent import AgentRunner
#
# async def process_files():
#     files = glob.glob(os.path.join("../../../ingestion-service/data", "*.pdf"))
#     initial_tools = []
#     print(f"fileSIze: f{len(files)}")
#     for file in files:
#         print(f"file: ${file}")
#         summary_tool, vector_tool = await create_doc_tools(file, Path(file).stem)  # 假设create_doc_tools是一个异步函数
#         initial_tools.append(summary_tool)
#         initial_tools.append(vector_tool)
#     print(str(initial_tools))
#     agent_worker = FunctionCallingAgentWorker.from_tools(
#         initial_tools,
#         llm=OpenAI(model="gpt-3.5-turbo"),
#         verbose=True
#     )
#     agent = AgentRunner(agent_worker)
#     response = agent.query("AI虚拟偶像的摘要")
#     print(f"response1: ${response.response}")
#     response = agent.chat("AI虚拟偶像的摘要")
#     print(f"response2: ${response.response}")
#
# async def main():
#     await process_files()
#
# if __name__ == "__main__":
#     print("asyncio!!")
#     asyncio.run(main())