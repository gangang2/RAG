from typing import List

from fastapi import APIRouter, HTTPException, status
from llama_index.core.llms import MessageRole
from pydantic import BaseModel
from nemoguardrails import LLMRails, RailsConfig
import os
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"
chat_router = r = APIRouter()

class _Message(BaseModel):
    role: MessageRole
    content: str

class _ChatData(BaseModel):
    messages: List[_Message]

@r.post("")
async def chat(data: _ChatData):
    print(f"len: ${len(data.messages)}")
    if len(data.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="没有提供消息",
        )
    lastMessage = data.messages.pop()
    if lastMessage.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前最后一个消息不是来自用户",
        )

    config = RailsConfig.from_path("E:\\data\\rag-microservices-agent\\rag-microservices\\inference-service\\backend\\app\\config")
    rails = LLMRails(config)

    prompt1 = "请用中文回答以下问题："
    prompt=prompt1 + lastMessage.content
    response = await rails.generate_async(prompt)

    return response
