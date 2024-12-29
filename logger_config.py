import logging
import logging.handlers
import os
from pathlib import Path
from filelock import FileLock
import shutil
import time
import atexit
import weakref

class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """线程安全的日志处理器"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock_file = str(Path(self.baseFilename).with_suffix('.lock'))
        self.file_lock = FileLock(self.lock_file, timeout=2)
        
    def emit(self, record):
        try:
            with self.file_lock:
                super().emit(record)
        except Exception:
            self.handleError(record)

    def close(self):
        try:
            self.file_lock.release(force=True)
        except:
            pass
        super().close()

_handlers = weakref.WeakValueDictionary()

# 全局日志锁
_log_locks = {}

def _cleanup_locks():
    """清理所有日志锁"""
    for lock in _log_locks.values():
        try:
            lock.release()
        except:
            pass

atexit.register(_cleanup_locks)

def clean_logs_directory():
    """安全清理日志目录"""
    log_dir = Path("logs")
    if not log_dir.exists():
        return
        
    # 1. 先关闭所有日志处理器
    for handler in _handlers.values():
        handler.close()
    
    # 2. 清理所有锁文件
    for path in log_dir.glob("*.lock"):
        try:
            path.unlink()
        except:
            pass
    
    # 3. 删除所有日志文件
    try:
        # 先尝试删除单个文件
        for file in log_dir.glob("*"):
            try:
                if file.is_file():
                    file.unlink(missing_ok=True)
            except:
                pass
        
        # 如果目录还存在但为空，则删除目录
        if log_dir.exists() and not any(log_dir.iterdir()):
            log_dir.rmdir()
    except Exception as e:
        print(f"Warning: Failed to clean some log files: {e}")
    
    # 4. 重新创建日志目录
    log_dir.mkdir(exist_ok=True)
    print(f"Cleaned logs directory: {log_dir}")

def setup_logger(name: str, log_file: str) -> logging.Logger:
    """线程安全的日志设置"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 使用安全的处理器
        if log_file not in _handlers:
            handler = SafeRotatingFileHandler(
                log_dir / log_file,
                maxBytes=10*1024*1024,
                backupCount=5,
                encoding='utf-8'
            )
            handler.setFormatter(formatter)
            _handlers[log_file] = handler
            
        logger.addHandler(_handlers[log_file])
        
    return logger
