import httpx
import json
import asyncio
from typing import AsyncGenerator, Dict, Any
import time
from logger_config import setup_logger
from utils.utils import wait_for_server

# 配置日志
logger = setup_logger(__name__, 'test_stream.log')

async def test_chat_stream() -> None:
    """测试聊天API的流式响应"""
    url = "http://localhost:8000/v1/chat/completions"
    
    # 测试消息
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "你好,请用中文回答:1+1等于几?"}
        ],
        "stream": True,
        "temperature": 0.7
    }
    
    print("发送请求...")
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream('POST', url, json=data, timeout=30.0) as response:
                response.raise_for_status()
                
                print("\n开始接收流式响应:")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            if "error" in chunk:
                                print(f"\n错误: {chunk['error']}")
                                break
                            
                            if chunk.get("choices", [{}])[0].get("finish_reason") == "stop":
                                break
                                
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                print(content, end="", flush=True)
                        except json.JSONDecodeError as e:
                            print(f"\n解析响应数据出错: {e}")
                            
        print(f"\n\n请求完成,耗时: {time.time() - start_time:.2f}秒")
        
    except httpx.HTTPStatusError as e:
        print(f"HTTP错误: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        print(f"请求错误: {e}")
    except Exception as e:
        print(f"其他错误: {e}")

async def main():
    """主函数"""
    print("开始测试流式聊天...")
    logger.info("开始测试流式聊天...")
    if await wait_for_server("http://localhost:8000/health"):
        await test_chat_stream()
    else:
        print("无法连接到服务器，测试终止")
        logger.error("无法连接到服务器，测试终止")

if __name__ == "__main__":
    asyncio.run(main())
