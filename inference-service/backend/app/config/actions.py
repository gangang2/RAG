# 需要更改的唯一文件是 actions.py ，我们在其中定义一个自定义操作以将 LlamaIndex 与 NeMo Guardrails 集成。
from pathlib import Path
from typing import Optional
from nemoguardrails.actions import action
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.llms.openai import OpenAI
from utils.util import create_doc_tools
from app.engine.index import get_index_and_query_engine

# from app.engine.index import get_index_and_query_engine


agent = None

import os
import glob

def get_pdf_files(directory):
    # 使用glob模块匹配data目录下所有.pdf文件
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    return pdf_files

async def init():
    global agent
    # logger = logging.getLogger(__name__)
    # logger.info("Initializing agent with tools")

    print(f"INIT!!!!!${os.getcwd()}")
    if agent is None:
        current_script_path = Path(__file__).absolute()
        parent_dir = current_script_path.parent.parent.parent.parent.parent
        print(parent_dir)
        data_dir_path = os.path.join(parent_dir, "ingestion-service", "data")
        files = get_pdf_files(data_dir_path)
        print(f"data_dir_path: ${data_dir_path} files: ${files}")
        # files = get_pdf_files("../../../../ingestion-service/data")
        print(str(files))
        initial_tools = []
        for file in files:
            absFile = os.path.abspath(file)
            print(str(absFile))
            summary_tool, vector_tool = await create_doc_tools(absFile, Path(file).stem)
            initial_tools.append(summary_tool)
            initial_tools.append(vector_tool)
        initial_tools.append(get_index_and_query_engine())
        print(str(initial_tools))

        agent_worker = FunctionCallingAgentWorker.from_tools(
            initial_tools,
            llm=OpenAI(model="gpt-3.5-turbo"),
            verbose=True
        )
        agent = AgentRunner(agent_worker)

    return agent

def get_query_response(agent: AgentRunner, query: str) -> str:
    """
    Function to query based on the query_engine and query string passed in.
    """
    response = agent.chat(query)
    response_str = response.response
    if response_str is None:
        return ""
    return response_str

@action(is_system_action=True)
async def user_query(context: Optional[dict] = None):
    """
    Function to invoke the query_engine to query user message and response.
    """
    user_message = context.get("user_message")
    print('==================================================================')
    print('==========user_message is ', user_message)
    query_engine = await init()
    return get_query_response(query_engine, user_message)

#
# if __name__ == '__main__':
#     # # 获取当前文件的绝对路径
#     # current_file_path = os.path.abspath(__file__)
#     #
#     # # 获取项目根目录的路径
#     # project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
#     # data_dir_path = os.path.join(project_root, "ingestion-service", "data")
#
#     # 获取当前脚本的绝对路径
#     current_script_path = Path(__file__).absolute()
#
#     # 获取当前脚本所在目录的上一级目录
#     parent_dir = current_script_path.parent.parent.parent.parent.parent
#     print(parent_dir)
#     data_dir_path = os.path.join(parent_dir, "ingestion-service", "data")
#     files = get_pdf_files(data_dir_path)
#     print(f"data_dir_path: ${data_dir_path} files: ${files}")
#     for file in files:
#         print(Path(file))
