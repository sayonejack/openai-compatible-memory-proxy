# openai-compatible-memory-proxy

基于FastAPI实现的OpenAI兼容聊天接口服务，集成了会话记忆管理，支持流式输出和多模型接入。支持 cline

## 功能特点

- OpenAI兼容的API接口
- 支持流式(SSE)和非流式响应
- 完整的日志系统
- 健康检查接口
- CORS支持
- 可配置的模型参数

## 安装

1. 克隆项目
```bash
git clone <repository-url>
cd <project-directory>
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 环境变量配置
   
项目根目录下提供了`.env.example`文件作为环境变量配置模板：

```bash
# API配置
OPENAI_API_KEY=your-api-key-here
HF_ENDPOINT=https://hf-mirror.com

# 服务配置
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# 模型配置
DEFAULT_MODEL=deepseek-chat
DEFAULT_TEMPERATURE=0.0
```

复制`.env.example`到`.env`文件：
```bash
cp .env.example .env
```

然后根据实际需求修改配置值。

4. 下载模型

项目提供了模型下载脚本，用于预先下载和缓存所需模型：

```bash
python download_models.py
```

支持的功能：
- 自动下载指定的模型文件
- 支持断点续传
- 可配置下载目录
- 显示下载进度
- 模型完整性验证

配置说明：
```bash
# .env 中的模型相关配置
HF_HOME=./models/hub           # 模型存储目录
HF_DATASETS_CACHE=./models/datasets  # 数据集缓存目录
MODEL_CACHE_DIR=./models/cache      # 模型缓存目录
```

## API接口

### 聊天完成接口

- 端点：`/v1/chat/completions`
- 方法：POST
- 请求格式：
```json
{
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": "你好"}
    ],
    "temperature": 0.0,
    "stream": true,
    "max_tokens": null
}
```

### 健康检查接口

- 端点：`/health`
- 方法：GET
- 响应示例：
```json
{
    "status": "healthy"
}
```

## 配置选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| model | 使用的模型名称 | deepseek-chat |
| temperature | 温度参数 | 0.0 |
| stream | 是否启用流式响应 | true |
| max_tokens | 最大生成token数 | null |

## 运行服务

```bash
python chat_api.py
```

服务将在 `http://0.0.0.0:8000` 启动

## 日志系统

- API日志：`logs/api.log`
- 支持自动日志清理
- 可配置的日志级别

## 贡献

欢迎提交Issue和Pull Request来帮助改进项目。

## 许可证

[待定] - 请指定适当的许可证
