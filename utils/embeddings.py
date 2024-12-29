import os
from pathlib import Path
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

class LocalEmbeddings(Embeddings):
    """使用本地模型的嵌入实现"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """初始化本地嵌入模型"""
        if cache_dir is None:
            cache_dir = os.getenv('MODEL_CACHE_DIR', './models/cache')
        
        # 使用Path处理路径
        cache_dir = Path(cache_dir).absolute()
        hub_dir = cache_dir.parent / 'hub'
        
        # 创建必要的目录
        cache_dir.mkdir(parents=True, exist_ok=True)
        hub_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置环境变量，使用绝对路径
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
        os.environ['HF_HOME'] = str(hub_dir)
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        try:
            # 首先尝试从本地加载
            local_model_path = cache_dir / model_name
            if local_model_path.exists():
                self.model = SentenceTransformer(str(local_model_path))
            else:
                # 如果本地没有，再尝试下载
                self.model = SentenceTransformer(
                    model_name_or_path=model_name,
                    cache_folder=str(cache_dir)
                )
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档嵌入"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """生成查询文本的嵌入"""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
