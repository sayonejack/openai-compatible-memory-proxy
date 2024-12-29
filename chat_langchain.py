from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage
)
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Literal
import os
from dotenv import load_dotenv
import logging
import json
import time
from pydantic import BaseModel, Field
from memory_store import MemoryStore
from logger_config import setup_logger

# 配置日志
logger = setup_logger(__name__, 'chat.log')

# Load environment variables // 加载环境变量
load_dotenv()

# Supported models // 支持的模型
SUPPORTED_MODELS = Literal[ "deepseek-chat"]

class ChatMessage(BaseModel):
    """Chat message model // 聊天消息模型"""
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    """Chat completion request model // 聊天补全请求模型"""
    model: SUPPORTED_MODELS = Field(..., description="Model to use for completion")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, gt=0)
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    """Chat completion response model // 聊天补全响应模型"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

    class Config:
        """Pydantic config // Pydantic配置"""
        json_schema_extra = {
            "example": {
                "id": "chat_1234567890",
                "object": "chat.completion",
                "created": 1677649420,
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "你好！有什么我可以帮你的吗？"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 56,
                    "completion_tokens": 78,
                    "total_tokens": 134
                }
            }
        }

class ChatBot:
    """Chat bot using LangChain // 使用LangChain的聊天机器人"""
    
    def __init__(self):
        """Initialize chat bot // 初始化聊天机器人"""
        logger.info("Initializing ChatBot")
        
        try:
            self.api_base = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")
            self.api_key = os.getenv("OPENAI_API_KEY")
            
            logger.debug(f"API base: {self.api_base}")
            logger.debug("API key found" if self.api_key else "API key not found")
            
            if not self.api_key:
                logger.error("OPENAI_API_KEY not found in environment variables")
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            logger.info(f"Initializing ChatBot with API base: {self.api_base}")
            
            # Initialize chat model // 初始化聊天模型
            try:
                logger.debug("Creating ChatOpenAI instance")
                self.chat = ChatOpenAI(
                    model_name="deepseek-chat",
                    openai_api_key=self.api_key,
                    openai_api_base=self.api_base,
                    streaming=True,
                    temperature=0.7,
                    request_timeout=60,  # 1 minute timeout // 1分钟超时
                    max_retries=2  # Reduce retries // 减少重试次数
                    # 删除不支持的 model_kwargs 参数
                )
                logger.info("ChatOpenAI instance created successfully")
            except Exception as model_error:
                logger.error(f"Error creating ChatOpenAI instance: {str(model_error)}", exc_info=True)
                raise ValueError(f"Failed to initialize chat model: {str(model_error)}")
            
            # Initialize memory store
            self.memory_store = MemoryStore()
            logger.info("Memory store initialized successfully")
            
            logger.info("ChatBot initialized successfully")
            logger.debug(f"ChatBot configuration: model={self.chat.model_name}, "
                        f"temperature={self.chat.temperature}, "
                        f"streaming={self.chat.streaming}, "
                        f"timeout={self.chat.request_timeout}")
                        
        except Exception as e:
            logger.error(f"Error initializing ChatBot: {str(e)}", exc_info=True)
            raise
    
    def _convert_message(self, message: ChatMessage) -> BaseMessage:
        """Convert message to LangChain message // 将消息转换为LangChain消息"""
        if message.role == "system":
            return SystemMessage(content=message.content)
        elif message.role == "user":
            return HumanMessage(content=message.content)
        elif message.role == "assistant":
            return AIMessage(content=message.content)
        else:
            raise ValueError(f"Unknown role: {message.role}")
    
    async def _process_conversation_with_memory(self, messages: List[ChatMessage]) -> List[BaseMessage]:
        """Process messages with memory retrieval"""
        try:
            current_msg = messages[-1].content
            relevant_memories = await self.memory_store.search_memory(current_msg)
            
            # 构建更结构化的系统消息
            system_messages = []
            if relevant_memories:
                memory_text = "\n".join([doc.page_content for doc in relevant_memories])
                system_messages.append(
                    ChatMessage(role="system", 
                              content="以下是之前对话的重要信息，请在回答时考虑这些信息，但不要完整重复你的自我介绍：\n" + memory_text)
                )
            
            # 添加行为指导
            system_messages.append(
                ChatMessage(role="system", 
                          content="请注意:\n1. 保持对话的连贯性\n2. 只在第一次交谈时做自我介绍\n"
                                 "3. 准确记住用户信息\n4. 在回答中自然地引用相关历史信息")
            )
            
            # 组合消息
            final_messages = system_messages + messages
            return [self._convert_message(m) for m in final_messages]
            
        except Exception as e:
            logger.error(f"Error processing conversation with memory: {str(e)}", exc_info=True)
            return [self._convert_message(m) for m in messages]

    async def _store_conversation(self, messages: List[ChatMessage], response: str):
        """Store conversation in memory with better context"""
        try:
            # 构建更有上下文的对话记录
            metadata = {
                "timestamp": time.time(),
                "context_type": "dialogue"
            }
            
            # 提取用户信息
            user_info = self._extract_user_info(messages[-1].content, response)
            if user_info:
                conversation = f"用户信息更新: {user_info}\n"
                metadata["context_type"] = "user_info"
            else:
                conversation = f"对话记录:\nUser: {messages[-1].content}\nAssistant: {response}"
            
            await self.memory_store.add_memory(
                text=conversation,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}", exc_info=True)

    def _extract_user_info(self, user_message: str, bot_response: str) -> Optional[str]:
        """提取用户信息"""
        info_markers = [
            "我叫", "我是", "我的名字", "我在", "我现在在",
            "我会", "我使用", "我掌握", "我学习"
        ]
        
        if any(marker in user_message for marker in info_markers):
            return f"{user_message} -> {bot_response}"
        return None

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming chat completion // 生成流式聊天补全"""
        try:
            logger.info("Processing streaming chat completion request")
            logger.debug(f"Request: {request.model_dump_json()}")
            
            # 处理记忆和消息
            langchain_messages = await self._process_conversation_with_memory(request.messages)
            
            # Update chat model parameters
            self.chat.model_name = request.model
            self.chat.temperature = request.temperature
            if request.max_tokens:
                self.chat.max_tokens = request.max_tokens
            self.chat.streaming = True
            
            logger.info("Starting streaming response")
            full_response = []  # 收集完整响应用于存储

            # 使用 astream 方法替代 agenerate，实现逐块处理
            async for chunk in self.chat.astream(langchain_messages):
                if not chunk.content:
                    continue
                    
                # 添加到完整响应
                full_response.append(chunk.content)
                
                # 立即返回当前块
                yield {
                    "id": f"chat_{hash(chunk.content)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": chunk.content
                        },
                        "finish_reason": None
                    }]
                }
            
            # 发送结束标记
            yield {
                "id": f"chat_{hash('done')}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            
            # 存储完整对话
            await self._store_conversation(
                request.messages, 
                "".join(full_response)
            )
            
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {str(e)}", exc_info=True)
            # 发送错误信息
            yield {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }

    async def chat_completion_sync(
        self,
        request: ChatCompletionRequest
    ) -> Dict[str, Any]:
        """Generate non-streaming chat completion // 生成非流式聊天补全"""
        try:
            logger.info("Processing non-streaming chat completion request")
            logger.debug(f"Request: {request.model_dump_json()}")
            
            # 处理记忆和消息
            langchain_messages = await self._process_conversation_with_memory(request.messages)
            
            # Update chat model parameters // 更新聊天模型参数
            logger.info("Updating chat model parameters")
            self.chat.model_name = request.model
            self.chat.temperature = request.temperature
            if request.max_tokens:
                self.chat.max_tokens = request.max_tokens
            self.chat.streaming = False
            
            logger.info(f"Using model: {self.chat.model_name} with temperature: {self.chat.temperature}")
            logger.info("Generating response")
            
            try:
                logger.debug("Calling chat.agenerate")
                response = await self.chat.agenerate([langchain_messages])
                generation = response.generations[0][0]
                
                # 存储对话
                await self._store_conversation(request.messages, generation.text)
                
                result = {
                    "id": "chat_" + str(hash(generation.text)),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generation.text
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": response.llm_output.get("token_usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": response.llm_output.get("token_usage", {}).get("completion_tokens", 0),
                        "total_tokens": response.llm_output.get("token_usage", {}).get("total_tokens", 0)
                    }
                }
                
                logger.info("Chat completion generated successfully")
                logger.debug(f"Response: {json.dumps(result, ensure_ascii=False)}")
                return result
                
            except Exception as api_error:
                logger.error(f"API call failed: {str(api_error)}", exc_info=True)
                logger.error(f"API base: {self.api_base}")
                logger.error(f"Model: {self.chat.model_name}")
                raise ValueError(f"API call failed: {str(api_error)}")
                
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
            raise

    async def chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Generate chat completion // 生成聊天补全"""
        logger.info("Entering chat_completion method")
        logger.debug(f"Request: {request.model_dump_json()}")
        logger.debug(f"Current chat model: {self.chat.model_name}, temperature: {self.chat.temperature}")
        
        try:
            if request.stream:
                logger.info("Using streaming mode")
                # Don't await the generator, just return it // 不要 await 生成器，直接返回它
                return self.chat_completion_stream(request)
            else:
                logger.info("Using non-streaming mode")
                return await self.chat_completion_sync(request)
        except Exception as e:
            logger.error(f"Error in chat_completion: {str(e)}", exc_info=True)
            raise

def main():
    """Main function for testing // 测试主函数"""
    import asyncio
    
    async def test():
        try:
            chatbot = ChatBot()
            
            # Test request // 测试请求
            request = ChatCompletionRequest(
                model="deepseek-chat",
                messages=[
                    ChatMessage(role="user", content="你好，请做个自我介绍")
                ],
                temperature=0.7,
                stream=False
            )
            
            # Test normal completion // 测试普通补全
            print("\nTesting normal completion:")
            try:
                response = await chatbot.chat_completion(request)
                print(json.dumps(response, ensure_ascii=False, indent=2))
            except Exception as e:
                logger.error(f"Error in normal completion: {str(e)}", exc_info=True)
                print(f"\nError in normal completion: {str(e)}")
            
            # Test streaming completion // 测试流式补全
            print("\nTesting streaming completion:")
            try:
                request.stream = True
                stream = await chatbot.chat_completion(request)
                async for chunk in stream:
                    if chunk["choices"][0].get("delta", {}).get("content"):
                        print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                    elif chunk["choices"][0].get("finish_reason") == "stop":
                        print("\nStream completed")
            except Exception as e:
                logger.error(f"Error in streaming completion: {str(e)}", exc_info=True)
                print(f"\nError in streaming completion: {str(e)}")
        except Exception as e:
            logger.error(f"Error in test: {str(e)}", exc_info=True)
            print(f"\nError in test: {str(e)}")
    
    # Run test // 运行测试
    asyncio.run(test())

if __name__ == "__main__":
    main()