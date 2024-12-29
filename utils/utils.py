import asyncio
import time
import aiohttp
import logging
from logger_config import setup_logger

# 配置日志
logger = setup_logger(__name__, 'utils.log')

async def wait_for_server(health_url: str, max_retries: int = 10, retry_interval: float = 5) -> bool:
    """等待服务器启动"""
    logger.info("等待服务器启动...")
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=5) as response:
                    if response.status == 200:
                        logger.info("服务器已就绪")
                        return True
                    logger.info(f"服务器未就绪，尝试次数：{attempt + 1}")
                    print(f"服务器未就绪，尝试次数：{attempt + 1}")
        except Exception as e:
            logger.warning(f"连接失败 ({attempt + 1}/{max_retries}): {str(e)}")
            print(f"等待服务器... 尝试 {attempt + 1}/{max_retries}")
        
        await asyncio.sleep(retry_interval)
    
    logger.error("服务器启动超时")
    print("服务器启动超时")
    return False