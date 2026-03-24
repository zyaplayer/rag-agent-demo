import os

from dotenv import load_dotenv

# 从项目根目录的 .env 加载环境变量到 os.environ；若文件不存在则静默跳过。
# 需在依赖中安装 python-dotenv（通常写入 requirements.txt）。
load_dotenv()


class Config:
    """
    全局配置命名空间（类属性即配置项）。
    设计为仅含类属性、无实例方法，便于 ``from config import Config`` 后
    以 ``Config.MILVUS_HOST`` 等形式访问，且便于测试时 monkeypatch。
    """

    # 大语言模型（DeepSeek，OpenAI 兼容 HTTP API）
    OPENAI_API_KEY = os.getenv("API")
    BASE_URL = "https://api.deepseek.com"
    MODEL_NAME = "deepseek-chat"

    # Milvus 向量数据库

    # gRPC / HTTP 服务地址；单机 Docker 部署时多为 localhost。
    MILVUS_HOST = "localhost"

    # Milvus 监听端口，默认 19530（与官方文档一致）。
    MILVUS_PORT = "19530"

    # 本 RAG 项目使用的集合（Collection）名称；建库、写入、检索需保持一致。
    COLLECTION_NAME = "rag_collection"
