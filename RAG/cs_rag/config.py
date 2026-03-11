import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """配置管理中心"""
    
    # 路径配置
    PROJECT_ROOT = Path(__file__).parent.absolute()
    KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
    VECTOR_INDEX_DIR = PROJECT_ROOT / "vector_index"
    LOGS_DIR = PROJECT_ROOT / "logs"
    METADATA_DIR = PROJECT_ROOT / "metadata"
    
    # 模型配置
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_PROVIDER = "ali_bailian"  # 阿里百炼平台
    LLM_MODEL_NAME = "qwen3.5-plus"
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 通过环境变量管理
    
    # RAG参数
    CHUNK_SIZE = 512  # 文本分块大小：512 tokens
    CHUNK_OVERLAP = 64  # 重叠长度：64 tokens
    RETRIEVAL_K = 5  # 检索数量：5个文档
    SIMILARITY_THRESHOLD = 0.7  # 相似度阈值：0.7
    
    # 提示词模板
    SYSTEM_PROMPT = """
    你是一个专业的网络安全专家，基于提供的知识库内容回答用户的问题。
    请确保回答准确、专业、简洁，并与提供的上下文信息保持一致。
    如果上下文信息不足以回答问题，请明确告知用户。
    """
    
    # 日志配置
    LOG_LEVEL = "INFO"  # 日志级别：Info
    LOG_FORMAT = "{time} | {level} | {message}"
    
    @classmethod
    def validate(cls):
        """验证配置的有效性"""
        if not cls.LLM_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY 环境变量未设置！请在环境变量中配置API密钥。")
        
        if not cls.KNOWLEDGE_BASE_DIR.exists():
            cls.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
            
        if not cls.VECTOR_INDEX_DIR.exists():
            cls.VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)
            
        if not cls.LOGS_DIR.exists():
            cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            
        if not cls.METADATA_DIR.exists():
            cls.METADATA_DIR.mkdir(parents=True, exist_ok=True)


# 初始化配置
try:
    Config.validate()
except ValueError as e:
    print(f"配置验证失败: {e}")
    exit(1)