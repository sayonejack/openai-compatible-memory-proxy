import aiohttp
import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional
from logger_config import setup_logger

# 配置日志（不清理目录，避免删除服务器的日志）
logger = setup_logger(__name__, 'test_api.log')

class TestResult:
    """测试结果统计"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.detailed_outputs = []  # 添加：存储详细输出
        
    def add_output(self, test_case: str, output: str):
        """添加测试用例的详细输出"""
        self.detailed_outputs.append(f"=== {test_case} ===\n{output}\n")
        
    def get_all_outputs(self) -> str:
        """获取所有详细输出"""
        return "\n".join(self.detailed_outputs)

    def add_result(self, case_name: str, success: bool, error_msg: str = None):
        """添加测试结果"""
        self.total += 1
        if success:
            self.passed += 1
        else:
            self.failed += 1
            if error_msg:
                self.errors.append(f"{case_name}: {error_msg}")

    def get_summary(self) -> str:
        """获取测试总结"""
        success_rate = (self.passed / self.total * 100) if self.total > 0 else 0
        summary = [
            "\n=== 测试结果总结 ===",
            f"总用例数: {self.total}",
            f"通过数: {self.passed}",
            f"失败数: {self.failed}",
            f"成功率: {success_rate:.1f}%"
        ]
        if self.errors:
            summary.extend(["\n失败详情:"] + self.errors)
        return "\n".join(summary)

    async def get_ai_summary(self, api_url: str) -> str:
        """使用API获取测试结果的AI总结"""
        messages = [
            {
                "role": "system",
                "content": "你是一个测试分析专家，请根据提供的测试结果生成专业的分析报告。"
            },
            {
                "role": "user",
                "content": f"""
请对以下测试执行结果进行专业分析并生成报告：

测试统计信息：
- 总用例数：{self.total}
- 通过数：{self.passed}
- 失败数：{self.failed}
- 成功率：{(self.passed / self.total * 100) if self.total > 0 else 0:.1f}%

详细测试过程：
{self.get_all_outputs()}

请从以下几个方面进行分析：
1. 测试覆盖情况
2. 成功用例分析
3. 失败用例分析（如果有）
4. 系统响应性能
5. 建议改进点

请生成一份简洁专业的总结报告。
"""
            }
        ]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{api_url}/v1/chat/completions",
                    json={
                        "model": "deepseek-chat",
                        "messages": messages,
                        "temperature": 0.3,
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        return f"获取AI分析报告失败: HTTP {response.status}"
        except Exception as e:
            return f"获取AI分析报告时发生错误: {str(e)}"

class ChatAPITest:
    def __init__(self, base_url: str = "http://localhost:8000", max_retries: int = 30, retry_interval: float = 1.0):
        self.base_url = base_url
        self.chat_url = f"{base_url}/v1/chat/completions"
        self.health_url = f"{base_url}/health"
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.test_results = TestResult()

    async def wait_for_server(self) -> bool:
        """等待服务器启动"""
        logger.info("等待服务器启动...")
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.health_url, timeout=5) as response:
                        if response.status == 200:
                            return True
                        logger.info(f"服务器就绪，尝试次数：{attempt + 1}")
                        print(f"服务器就绪，尝试次数：{attempt + 1}")
            except aiohttp.ClientError as e:
                logger.warning(f"连接失败 ({attempt + 1}/{self.max_retries}): {str(e)}")
            except asyncio.TimeoutError:
                logger.warning(f"连接超时 ({attempt + 1}/{self.max_retries})")
            except Exception as e:
                logger.error(f"未知错误 ({attempt + 1}/{self.max_retries}): {str(e)}")
            
            print(f"等待服务器... 尝试 {attempt + 1}/{self.max_retries}")
            await asyncio.sleep(self.retry_interval)
        
        logger.error("服务器启动超时")
        print("服务器启动超时")
        return False

    async def test_health(self) -> bool:
        """测试健康检查接口"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.health_url) as response:
                    result = await response.json()
                    print(f"Health check result: {result}")
                    return result.get("status") == "healthy"
        except Exception as e:
            print(f"Health check failed: {str(e)}")
            return False

    async def chat_completion(self, case_name: str, messages: List[Dict[str, str]]) -> bool:
        """测试流式聊天补全接口"""
        request_data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "stream": True
        }

        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                output_buffer = []  # 添加：用于收集输出
                
                async with session.post(self.chat_url, json=request_data) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        self.test_results.add_result(case_name, False, f"HTTP {response.status}: {error_msg}")
                        return False

                    print(f"\n=== {case_name} ===")
                    case_output = f"Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}\n\nResponse:\n"
                    output_buffer.append(case_output)
                    print(case_output)
                    
                    received_content = False
                    full_response = []  # 添加：收集完整响应
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                if "error" in data:
                                    self.test_results.add_result(case_name, False, str(data["error"]))
                                    return False
                                if "choices" in data and data["choices"]:
                                    content = data["choices"][0].get("delta", {}).get("content", "")
                                    if content:
                                        received_content = True
                                        full_response.append(content)  # 收集响应
                                        print(content, end="", flush=True)
                            except json.JSONDecodeError:
                                continue

                    end_time = time.time()
                    
                    # 添加响应到输出缓冲
                    response_text = "".join(full_response)
                    output_buffer.append(response_text)
                    
                    timing_info = f"\n请求用时: {end_time - start_time:.2f} 秒\n"
                    output_buffer.append(timing_info)
                    print(timing_info)
                    
                    # 保存完整输出到测试结果
                    self.test_results.add_output(case_name, "".join(output_buffer))
                    
                    if not received_content:
                        self.test_results.add_result(case_name, False, "未收到响应内容")
                        return False
                    
                    self.test_results.add_result(case_name, True)
                    return True

        except Exception as e:
            error_msg = f"请求执行错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.test_results.add_result(case_name, False, error_msg)
            return False

async def main():
    try:
        logger.info("开始测试")
        print("\n=== 开始测试 ===")
        
        tester = ChatAPITest(
            max_retries=10,
            retry_interval=5.0
        )

        # 等待服务器启动
        if not await tester.wait_for_server():
            print("服务器未就绪，退出测试")
            return

        # 健康检查
        if not await tester.test_health():
            logger.error("健康检查失败，退出测试")
            print("健康检查失败，退出测试")
            return

        # 增强的测试用例集
        test_cases = [
            ("基础对话", [
                {"role": "user", "content": "你好，请介绍一下你自己"}
            ]),
            
            ("基础记忆测试", [
                {"role": "user", "content": "我叫小明"}
            ]),
            
            ("简单记忆回顾", [
                {"role": "user", "content": "还记得我的名字吗？"}
            ]),
            
            ("上下文记忆测试", [
                {"role": "user", "content": "我现在在北京工作"},
                {"role": "assistant", "content": "明白了，你在北京工作。"},
                {"role": "user", "content": "能完整说出我的信息吗？包括名字和工作地点"}
            ]),
            
            ("复杂信息记忆", [
                {"role": "user", "content": "我是一名程序员，主要使用Python和JavaScript"},
                {"role": "assistant", "content": "了解，你是使用Python和JavaScript的程序员。"},
                {"role": "user", "content": "我还会Rust语言"},
                {"role": "assistant", "content": "好的，你还会Rust语言。"},
                {"role": "user", "content": "请总结一下我会的编程语言有哪些"}
            ]),
            
            ("长期记忆测试", [
                {"role": "user", "content": "让我们聊个新话题。你觉得人工智能未来会如何发展？"}
            ]),
            
            ("记忆混合测试", [
                {"role": "user", "content": "现在请回顾一下我们之前聊过的所有内容，包括我的个人信息和我们讨论过的话题"}
            ]),
            
            ("专业知识应用", [
                {"role": "user", "content": "考虑到我会的编程语言，你觉得我更适合做前端还是后端开发？请解释原因"}
            ]),
            
            ("记忆清晰度测试", [
                {"role": "user", "content": "请按时间顺序列出我们整个对话中你了解到的关于我的所有信息"}
            ])
        ]

        # 运行测试用例
        print("\n=== 开始执行记忆测试用例 ===")
        print("注意：测试用例设计为连续对话，检验模型的记忆能力")
        print("每个测试用例都基于之前对话的积累\n")

        for i, (case_name, messages) in enumerate(test_cases, 1):
            print(f"\n=== 测试用例 {i}: {case_name} ===")
            if not await tester.chat_completion(case_name, messages):
                logger.error(f"测试用例 {case_name} 失败")
                print(f"\n测试用例 {case_name} 失败")
                print("由于测试用例之间存在依赖关系，停止后续测试")
                break
            print("\n" + "="*50)
            await asyncio.sleep(1)  # 添加间隔，避免太快

        # 打印测试总结
        print("\n" + "="*30 + " 测试结果 " + "="*30)
        print(tester.test_results.get_summary())
        
        if tester.test_results.failed == 0:
            print("\n✅ 所有记忆测试用例执行成功！")
        else:
            print(f"\n❌ 有 {tester.test_results.failed} 个测试用例失败")
            print("请查看上方日志获取详细信息")

        # 运行完所有测试后，获取AI总结
        print("\n=== 正在生成AI测试分析报告 ===")
        ai_summary = await tester.test_results.get_ai_summary(tester.base_url)
        
        print("\n=== AI 测试分析报告 ===")
        print(ai_summary)
        
        # 保存完整报告到文件
        report_path = "test_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== 测试执行详情 ===\n")
            f.write(tester.test_results.get_all_outputs())
            f.write("\n=== 测试统计 ===\n")
            f.write(tester.test_results.get_summary())
            f.write("\n\n=== AI 分析报告 ===\n")
            f.write(ai_summary)
        
        print(f"\n完整测试报告已保存到: {report_path}")

    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
        print(f"\n测试过程中发生错误: {str(e)}")
        print("请查看 test_api.log 获取详细错误信息")

if __name__ == "__main__":
    try:
        print("正在启动测试...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"程序异常退出: {str(e)}")
        logger.error("程序异常退出", exc_info=True)
