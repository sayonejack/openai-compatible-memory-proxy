import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import shutil

def download_model():
    """下载并保存模型到本地"""
    model_name = "all-MiniLM-L6-v2"
    cache_dir = Path('./models/cache').absolute()
    
    print(f"开始下载模型 {model_name}")
    print(f"缓存目录: {cache_dir}")
    
    try:
        # 确保目录存在
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 临时禁用离线模式来下载模型
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        os.environ.pop('HF_HUB_OFFLINE', None)
        
        # 下载模型
        model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        
        # 保存模型到指定目录
        model_path = cache_dir / model_name
        model.save(str(model_path))
        
        print(f"模型下载成功: {model_path}")
        return True
        
    except Exception as e:
        print(f"模型下载失败: {str(e)}")
        return False

if __name__ == "__main__":
    download_model()
