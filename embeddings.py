from typing import List
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

class LocalEmbeddings(Embeddings):
    """使用本地模型的嵌入实现"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """初始化本地嵌入模型"""
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档嵌入"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """生成查询文本的嵌入"""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
