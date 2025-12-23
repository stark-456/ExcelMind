"""知识库管理模块 - 支持灵活的知识导入和检索"""

import hashlib
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import yaml
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

from .config import get_config


@dataclass
class KnowledgeItem:
    """知识条目"""
    id: str
    content: str
    title: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    related_columns: List[str] = field(default_factory=list)
    priority: str = "normal"
    source_file: str = ""


class KnowledgeBase:
    """知识库管理类"""
    
    def __init__(self):
        config = get_config()
        self.kb_config = config.knowledge_base
        self.emb_config = config.embedding
        
        # 使用 OpenAI 兼容的 Embedding API
        self.embeddings = OpenAIEmbeddings(
            model=self.emb_config.model,
            openai_api_key=self.emb_config.api_key,
            openai_api_base=self.emb_config.api_url,
        )
        
        # 初始化 Chroma
        self.client = chromadb.PersistentClient(
            path=self.kb_config.vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="excel_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
    
    def load_from_file(self, file_path: Path) -> KnowledgeItem:
        """从文件加载知识条目（支持多种格式）"""
        content = file_path.read_text(encoding="utf-8")
        
        # 尝试解析 YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    metadata = yaml.safe_load(parts[1])
                    body = parts[2].strip()
                    return self._create_item_with_metadata(metadata, body, file_path)
                except yaml.YAMLError:
                    pass
        
        # 无 frontmatter 或解析失败，自动提取元数据
        return self._create_item_auto(content, file_path)
    
    def _create_item_with_metadata(self, metadata: dict, body: str, source: Path) -> KnowledgeItem:
        """使用显式元数据创建知识条目"""
        # 生成 ID（优先使用元数据中的 ID）
        item_id = metadata.get("id")
        if not item_id:
            file_hash = hashlib.md5(body.encode()).hexdigest()[:8]
            item_id = f"kb_{source.stem}_{file_hash}"
        
        return KnowledgeItem(
            id=item_id,
            content=body,
            title=metadata.get("title", source.stem),
            category=metadata.get("category", "general"),
            tags=metadata.get("tags", []),
            related_columns=metadata.get("related_columns", []),
            priority=metadata.get("priority", "normal"),
            source_file=str(source)
        )
    
    def _create_item_auto(self, content: str, source: Path) -> KnowledgeItem:
        """自动创建知识条目（无结构化元数据）"""
        # 生成 ID
        file_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        item_id = f"kb_{source.stem}_{file_hash}"
        
        # 提取标题（第一行或前50字）
        lines = content.strip().split("\n")
        title = lines[0].lstrip("#").strip()[:50] if lines else "未命名"
        
        # 提取关键词作为 tags（简单实现：提取中文词汇）
        words = re.findall(r'[\u4e00-\u9fa5]+', content)
        tags = list(set([w for w in words if len(w) >= 2]))[:10]
        
        return KnowledgeItem(
            id=item_id,
            content=content,
            title=title,
            tags=tags,
            source_file=str(source)
        )
    
    def search(self, query: str, columns: List[str] = None, top_k: int = None) -> List[KnowledgeItem]:
        """检索相关知识
        
        Args:
            query: 查询文本
            columns: 可选，当前表的列名列表，用于过滤相关列
            top_k: 返回结果数量
        """
        if top_k is None:
            top_k = self.kb_config.top_k
        
        # 检查集合是否为空
        if self.collection.count() == 0:
            return []
        
        # 生成查询向量
        query_embedding = self.embeddings.embed_query(query)
        
        # 向量检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        items = []
        if not results["documents"] or not results["documents"][0]:
            return items
            
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            # 过滤低相似度结果（余弦距离，越小越相似）
            if distance > (1 - self.kb_config.similarity_threshold):
                continue
            
            items.append(KnowledgeItem(
                id=meta.get("id", ""),
                content=doc,
                title=meta.get("title", ""),
                category=meta.get("category", "general"),
                tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                related_columns=meta.get("related_columns", "").split(",") if meta.get("related_columns") else [],
                priority=meta.get("priority", "normal"),
                source_file=meta.get("source_file", "")
            ))
        
        return items
    
    def add_entry(self, item: KnowledgeItem) -> None:
        """添加知识条目到向量库"""
        embedding = self.embeddings.embed_query(item.content)
        
        self.collection.upsert(
            ids=[item.id],
            embeddings=[embedding],
            documents=[item.content],
            metadatas=[{
                "id": item.id,
                "title": item.title,
                "category": item.category,
                "tags": ",".join(item.tags) if item.tags else "",
                "related_columns": ",".join(item.related_columns) if item.related_columns else "",
                "priority": item.priority,
                "source_file": item.source_file
            }]
        )
    
    def delete_entry(self, item_id: str) -> bool:
        """删除知识条目"""
        try:
            self.collection.delete(ids=[item_id])
            return True
        except Exception:
            return False
    
    def update_entry(self, item_id: str, content: str = None, title: str = None, 
                     category: str = None, tags: List[str] = None) -> bool:
        """更新知识条目（支持部分更新）"""
        # 获取现有条目
        results = self.collection.get(ids=[item_id], include=["documents", "metadatas"])
        if not results["ids"]:
            return False
        
        old_doc = results["documents"][0]
        old_meta = results["metadatas"][0]
        
        # 合并更新
        new_content = content if content is not None else old_doc
        new_meta = {
            "id": item_id,
            "title": title if title is not None else old_meta.get("title", ""),
            "category": category if category is not None else old_meta.get("category", "general"),
            "tags": ",".join(tags) if tags is not None else old_meta.get("tags", ""),
            "related_columns": old_meta.get("related_columns", ""),
            "priority": old_meta.get("priority", "normal"),
            "source_file": old_meta.get("source_file", "")
        }
        
        # 重新生成向量（如果内容变化）
        embedding = self.embeddings.embed_query(new_content)
        
        self.collection.upsert(
            ids=[item_id],
            embeddings=[embedding],
            documents=[new_content],
            metadatas=[new_meta]
        )
        return True
    
    def list_entries(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """列出所有知识条目（分页）"""
        results = self.collection.get(
            include=["metadatas"],
            limit=limit,
            offset=offset
        )
        
        entries = []
        for i, item_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            entries.append({
                "id": item_id,
                "title": meta.get("title", ""),
                "category": meta.get("category", "general"),
                "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                "source_file": meta.get("source_file", "")
            })
        
        return entries
    
    def get_entry(self, item_id: str) -> Optional[KnowledgeItem]:
        """获取单个知识条目详情"""
        results = self.collection.get(
            ids=[item_id],
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            return None
        
        doc = results["documents"][0]
        meta = results["metadatas"][0]
        
        return KnowledgeItem(
            id=item_id,
            content=doc,
            title=meta.get("title", ""),
            category=meta.get("category", "general"),
            tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
            related_columns=meta.get("related_columns", "").split(",") if meta.get("related_columns") else [],
            priority=meta.get("priority", "normal"),
            source_file=meta.get("source_file", "")
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        count = self.collection.count()
        return {
            "total_entries": count,
            "vector_db_path": self.kb_config.vector_db_path,
            "embedding_model": self.emb_config.model
        }
    
    def index_directory(self, dir_path: Path = None) -> int:
        """索引目录下所有知识文件"""
        if dir_path is None:
            dir_path = Path(self.kb_config.knowledge_dir)
        
        if not dir_path.exists():
            return 0
        
        count = 0
        for file_path in dir_path.rglob("*"):
            if file_path.suffix in [".md", ".txt", ".markdown"]:
                item = self.load_from_file(file_path)
                self.add_entry(item)
                count += 1
        
        return count


# 全局实例
_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base() -> Optional[KnowledgeBase]:
    """获取知识库实例"""
    global _knowledge_base
    config = get_config()
    
    if not config.knowledge_base.enabled:
        return None
    
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    
    return _knowledge_base


def reset_knowledge_base() -> None:
    """重置知识库实例"""
    global _knowledge_base
    _knowledge_base = None


def format_knowledge_context(items: List[KnowledgeItem]) -> str:
    """将知识条目格式化为可注入 Prompt 的文本"""
    if not items:
        return "暂无相关知识参考。"
    
    parts = []
    for i, item in enumerate(items, 1):
        parts.append(f"### 参考知识 {i}: {item.title}")
        parts.append(item.content)
        if item.tags:
            parts.append(f"**标签**: {', '.join(item.tags)}")
        parts.append("")
    
    return "\n".join(parts)
