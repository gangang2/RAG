安装环境
pip install uvicorn fastapi fastapi[all] nemoguardrails pymilvus uvicorn python-dotenv llama-index-vector-stores-milvus llama-index llama-index-postprocessor-cohere-rerank llama-index-embeddings-openai llama-index-embeddings-huggingface llama-index-llms-openai

这是一个简易的rag系统
使用LLamaIndex作为RAG的数据框架 与Langchain相似 相比于langchain更专注于数据处理
使用Uvicorn高性能服务器运行应用程序
使用FastAPI高性能的web 框架来定义路由、处理请求和响应。
使用英伟达的NeMoGuardRails的开源框架作为安全审查，通过控制输入和输出，避免生成有风险的内容
使用milvus作为向量数据库
使用cohere来对检索结果进行ReRank重排获取前N个最优文档实现RAG检索增强
使用next.js构建React应用程序
