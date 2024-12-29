# export HF_ENDPOINT=https://hf-mirror.com
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from chat_langchain import ChatBot, ChatCompletionRequest, ChatMessage
import asyncio
import logging
from logger_config import setup_logger, clean_logs_directory

# 清理日志目录
clean_logs_directory()

# 配置日志
logger = setup_logger(__name__, 'api.log')

# 设置第三方库的日志级别
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# 在FastAPI应用定义前添加配置类
class ChatConfig:
    """聊天配置类"""
    def __init__(
        self,
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        stream: bool = False,
        stream_options: dict = None,
        max_tokens: int = None,
        api_key: str = None
    ):
        self.model = model
        self.temperature = temperature
        self.stream = stream
        self.stream_options = stream_options
        self.max_tokens = max_tokens
        self.api_key = api_key
    
    @classmethod
    def from_request(cls, request: Request, request_data: dict) -> "ChatConfig":
        """从请求数据创建配置"""
        # 从请求头中获取API key
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
        logger.info(f"API key: {api_key}")

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")  # 如果请求头中没有，则使用环境变量

        return cls(
            model=request_data.get("model", "deepseek-chat"),
            temperature=request_data.get("temperature", 0.0),
            stream=request_data.get("stream", True),
            stream_options=request_data.get("stream_options"),
            max_tokens=request_data.get("max_tokens"),
            api_key=api_key
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "stream": self.stream,
            "stream_options": self.stream_options,
            "max_tokens": self.max_tokens
        }

app = FastAPI(title="Chat API", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化ChatBot
chatbot = ChatBot()

async def stream_response(request: Request, request_data: dict):
    """处理流式响应"""
    try:
        logger.info(f"Stream request data: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
        
        # 处理配置和消息
        config = ChatConfig.from_request(request, request_data)
        messages = process_messages(request_data["messages"])
        
        logger.info(f"Using config: {config.to_dict()}")
        # 修复：使用列表转换而不是列表推导式
        processed_messages = [{"role": m.role, "content": m.content} for m in messages]
        logger.info(f"Processed messages: {json.dumps(processed_messages, ensure_ascii=False)}")
        
        chat_request = ChatCompletionRequest(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True
        )
        
        # 获取流式响应
        async_gen = await chatbot.chat_completion(chat_request)
        
        async for chunk in async_gen:
            # 转换为SSE格式
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        logger.error(f"Stream response error: {str(e)}", exc_info=True)
        error_response = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"

def process_messages(messages: list) -> list:
    """处理消息格式"""
    processed = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            text_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content.append(item.get("text", ""))
            content = " ".join(text_content)
        processed.append(ChatMessage(role=msg["role"], content=content))
    return processed

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI 兼容的聊天补全接口"""
    try:
        request_data = await request.json()
        logger.info(f"Raw request headers: {str(request.headers)}")
        logger.info(f"Raw request data: {json.dumps(request_data, ensure_ascii=False)}")
        
        # 处理配置和消息
        config = ChatConfig.from_request(request, request_data)
        messages = process_messages(request_data["messages"])
        
        logger.info(f"Using config: {config.to_dict()}")
        # 修复：使用列表转换而不是列表推导式
        processed_messages = [{"role": m.role, "content": m.content} for m in messages]
        logger.info(f"Processed messages: {json.dumps(processed_messages, ensure_ascii=False)}")
        
        # 检查是否为流式请求
        if config.stream:
            return StreamingResponse(
                stream_response(request, request_data),
                media_type="text/event-stream"
            )
        
        # 非流式请求处理
        chat_request = ChatCompletionRequest(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=False
        )
        
        response = await chatbot.chat_completion(chat_request)
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }
        )

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Initializing API server...")
    uvicorn.run(
        "chat_api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        reload_dirs=["D:/Models/openai/test2"],  # 限制监视目录
        log_level="info",  # 设置 uvicorn 日志级别
        reload_delay=1  # 增加重载延迟，减少文件系统事件
    )
