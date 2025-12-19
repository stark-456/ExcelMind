"""配置管理模块"""

import os
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """模型配置"""
    provider: str = "openai"
    model_name: str = "qwen3-80b"
    api_key: str = ""
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096


class ExcelConfig(BaseModel):
    """Excel 配置"""
    max_preview_rows: int = 5
    default_result_limit: int = 20
    max_result_limit: int = 1000


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置"""
    model: str = "qwen3-embedding-0.6b"
    dims: int = 1536
    api_url: str = "http://10.62.36.7:13206/member1/qwen3-embedding/v1"
    api_key: str = "empty"


class KnowledgeBaseConfig(BaseModel):
    """知识库配置"""
    enabled: bool = True
    knowledge_dir: str = "knowledge"  # 知识条目目录
    vector_db_path: str = ".vector_db"  # 向量库持久化路径
    top_k: int = 3  # 检索返回条数
    similarity_threshold: float = 0.7  # 相似度阈值
    auto_extract_metadata: bool = True  # 是否自动提取元数据


class AppConfig(BaseModel):
    """应用配置"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    excel: ExcelConfig = Field(default_factory=ExcelConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)


def _expand_env_vars(value: str) -> str:
    """展开环境变量 ${VAR} 格式"""
    pattern = re.compile(r'\$\{(\w+)\}')
    
    def replacer(match):
        env_var = match.group(1)
        return os.environ.get(env_var, "")
    
    return pattern.sub(replacer, value)


def _process_config_dict(config: dict) -> dict:
    """递归处理配置字典中的环境变量"""
    result = {}
    for key, value in config.items():
        if isinstance(value, str):
            result[key] = _expand_env_vars(value)
        elif isinstance(value, dict):
            result[key] = _process_config_dict(value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为项目根目录的 config.yaml
        
    Returns:
        AppConfig 实例
    """
    if config_path is None:
        # 查找配置文件
        possible_paths = [
            Path("config.yaml"),
            Path(__file__).parent.parent.parent / "config.yaml",
        ]
        for p in possible_paths:
            if p.exists():
                config_path = str(p)
                break
    
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}
        config_dict = _process_config_dict(raw_config)
        return AppConfig(**config_dict)
    
    return AppConfig()


# 全局配置实例
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: AppConfig) -> None:
    """设置全局配置实例"""
    global _config
    _config = config
