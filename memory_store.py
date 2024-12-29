from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings  # 更新导入
from langchain_community.chat_message_histories import ChatMessageHistory  # 更新导入路径
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
from utils.embeddings import LocalEmbeddings
import os
import time
import asyncio
import random
import logging
from logger_config import setup_logger
import hashlib
import json
from pathlib import Path

# 配置日志
logger = setup_logger(__name__, 'memory.log')

# 设置 chromadb 日志级别
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry").setLevel(logging.WARNING)

class MemoryStore:
    _instance = None
    _lock = asyncio.Lock()
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def __ainit__(self):
        """异步初始化"""
        if self._initialized:
            return
            
        async with self._lock:
            if self._initialized:
                return
                
            self.embeddings = LocalEmbeddings()
            logger.info("Local embeddings model initialized")
            
            # 初始化向量存储
            self.vectorstore = Chroma(
                collection_name="chat_memory",
                embedding_function=self.embeddings
            )
            
            self._initialized = True
    
    def __init__(self):
        """同步初始化基本属性"""
        if not hasattr(self, 'memory_types'):
            self.memory_types = {
                "personal_info": ["我叫", "我是", "我的名字"],
                "skills": ["我会", "我使用", "我掌握"],
                "location": ["在", "位于", "工作地点"],
                "dialogue": []
            }
            self.message_history = ChatMessageHistory()

    def __init__(self, persist_directory: str = "data/memory", 
                 cache_dir: str = "data/cache"):
        if self._initialized:
            return
            
        self.persist_directory = persist_directory
        self.cache_dir = cache_dir
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化本地嵌入模型（延迟初始化）
        self.embeddings = None
        self._embedding_lock = asyncio.Lock()
        
        # 更新 Chroma 客户端设置，只使用支持的参数
        self.client_settings = ChromaSettings(
            anonymized_telemetry=False,
            persist_directory=persist_directory,
            is_persistent=True
        )
        
        # 延迟初始化向量存储
        self.vectorstore = None
        self.retriever = None
        
        self._initialized = True
        
        # 改用消息历史记录
        self.message_history = ChatMessageHistory()

        # 定义记忆类型
        self.memory_types = {
            "personal_info": ["我叫", "我是", "我的名字"],
            "skills": ["我会", "我使用", "我掌握"],
            "location": ["在", "位于", "工作地点"],
            "dialogue": []
        }

    async def _ensure_initialized(self):
        """确保模型和存储已初始化"""
        if self.embeddings is None:
            async with self._embedding_lock:
                if self.embeddings is None:  # 双重检查
                    self.embeddings = LocalEmbeddings()
                    logger.info("Local embeddings model initialized")
                    
                    # 初始化向量存储，添加额外的错误处理
                    try:
                        self.vectorstore = Chroma(
                            collection_name="chat_memory",
                            embedding_function=self.embeddings,
                            client_settings=self.client_settings,
                            persist_directory=self.persist_directory  # 显式指定持久化目录
                        )
                        
                        self.retriever = self.vectorstore.as_retriever(
                            search_kwargs={"k": 10}
                        )
                    except Exception as e:
                        logger.error(f"Error initializing Chroma: {str(e)}")
                        # 使用基本配置重试
                        self.vectorstore = Chroma(
                            collection_name="chat_memory",
                            embedding_function=self.embeddings
                        )
                        self.retriever = self.vectorstore.as_retriever(
                            search_kwargs={"k": 10}
                        )

    def _get_cache_path(self, text: str) -> Path:
        """获取缓存文件路径"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return Path(self.cache_dir) / f"{text_hash}.json"

    async def _get_embedding_with_cache(self, text: str) -> Optional[List[float]]:
        """获取文本嵌入向量，使用缓存"""
        await self._ensure_initialized()
        cache_path = self._get_cache_path(text)
        
        # 检查缓存
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    logger.info(f"Using cached embedding for: {text[:50]}...")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        try:
            # 生成嵌入向量
            embedding = self.embeddings.embed_query(text)
            
            # 保存到缓存
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(embedding, f)
                logger.info(f"Cached new embedding for: {text[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
            
            return embedding
                
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None

    async def _classify_memory(self, text: str) -> str:
        """识别记忆类型"""
        text_lower = text.lower()
        for memory_type, keywords in self.memory_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return memory_type
        return "dialogue"

    async def add_memory(self, text: str, metadata: Dict[str, Any] = None):
        """改进的记忆添加"""
        await self._ensure_initialized()
        try:
            if metadata is None:
                metadata = {}
            
            # 增强元数据
            memory_type = await self._classify_memory(text)
            metadata.update({
                "timestamp": time.time(),
                "type": memory_type,
                "dialogue_id": str(int(time.time() * 1000))  # 用于关联对话
            })
            
            embedding = await self._get_embedding_with_cache(text)
            if embedding is None:
                logger.warning("Skipping memory addition due to embedding failure")
                return False
            
            document = Document(
                page_content=text,
                metadata=metadata
            )
            
            self.vectorstore.add_documents([document])
            logger.info(f"Added memory with type {memory_type}: {text[:50]}...")
            
            # 添加到消息历史
            if metadata.get("role") == "user":
                self.message_history.add_user_message(text)
            elif metadata.get("role") == "assistant":
                self.message_history.add_ai_message(text)
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding memory: {str(e)}", exc_info=True)
            return False

    async def search_memory(self, query: str, k: int = 10) -> List[Document]:
        """改进的记忆搜索策略"""
        await self._ensure_initialized()
        try:
            embedding = await self._get_embedding_with_cache(query)
            if embedding is None:
                return []
            
            # 分类搜索
            results = []
            for memory_type in self.memory_types.keys():
                type_results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter={"type": memory_type}
                )
                results.extend(type_results)
            
            # 结果排序和去重
            seen = set()
            unique_results = []
            for doc in sorted(results, key=lambda x: x.metadata.get("timestamp", 0)):
                content = doc.page_content
                if content not in seen:
                    seen.add(content)
                    unique_results.append(doc)
            
            logger.info(f"Retrieved {len(unique_results)} unique memories")
            return unique_results[:k]
            
        except Exception as e:
            logger.error(f"Error searching memory: {str(e)}", exc_info=True)
            return []

    def get_relevant_memories(self, query: str) -> str:
        """改进的记忆获取方法"""
        try:
            results = self.vectorstore.similarity_search(
                query,
                k=5,
                filter=None
            )
            return "\n".join(doc.page_content for doc in results)
        except Exception as e:
            logger.error(f"Error getting relevant memories: {str(e)}", exc_info=True)
            return ""
