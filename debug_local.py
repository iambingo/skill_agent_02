"""
本地 Debug 脚本 —— 绕过 Dify 框架直接运行 SkillAgentTool
=============================================================
支持两种 LLM 后端（通过环境变量控制）：
  Anthropic:        ANTHROPIC_API_KEY=sk-ant-...
  OpenAI 兼容接口:  OPENAI_API_KEY=...  OPENAI_BASE_URL=https://...  OPENAI_MODEL=...

用法示例：
  # 用 Anthropic
  ANTHROPIC_API_KEY=sk-ant-xxx python debug_local.py

  # 用 OpenAI/兼容接口（如 DeepSeek、本地 Ollama 等）
  OPENAI_API_KEY=sk-xxx OPENAI_BASE_URL=https://api.deepseek.com OPENAI_MODEL=deepseek-chat python debug_local.py

可修改底部 run_debug() 中的参数来切换 query / skill_name 等。
"""

import json
import os
import sys
import uuid
from typing import Any, Generator

# ── 确保项目根目录在 path 里 ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Dify 消息类型（真实的，无需 mock）────────────────────────────────────────
from dify_plugin.entities.model.llm import (
    LLMModelConfig,
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMUsage,
    ModelType,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageRole,
    PromptMessageTool,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.tool import ToolInvokeMessage


# ══════════════════════════════════════════════════════════════════════════════
# 1. Mock Storage（内存 dict）
# ══════════════════════════════════════════════════════════════════════════════
class MockStorage:
    def __init__(self):
        self._data: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    def set(self, key: str, value: bytes) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 把 Dify PromptMessage 列表转成各 SDK 的 messages 格式
# ══════════════════════════════════════════════════════════════════════════════
def _dify_msg_to_openai(msg: PromptMessage) -> dict:
    """转换单条 Dify 消息 → OpenAI messages 格式"""
    role_map = {
        PromptMessageRole.SYSTEM: "system",
        PromptMessageRole.USER: "user",
        PromptMessageRole.ASSISTANT: "assistant",
        PromptMessageRole.TOOL: "tool",
    }
    role = role_map.get(msg.role, "user")
    content = msg.content or ""

    if isinstance(msg, ToolPromptMessage):
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
        }

    if isinstance(msg, AssistantPromptMessage):
        result: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                        if isinstance(tc.function.arguments, str)
                        else json.dumps(tc.function.arguments, ensure_ascii=False),
                    },
                }
                for tc in msg.tool_calls
            ]
        return result

    return {"role": role, "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)}


def _dify_tools_to_openai(tools: list[PromptMessageTool] | None) -> list[dict] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _openai_response_to_dify_chunk(chunk_data: dict) -> LLMResultChunk | None:
    """把 OpenAI streaming chunk 转成 LLMResultChunk"""
    choices = chunk_data.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    finish_reason = choices[0].get("finish_reason")

    text_content = delta.get("content") or ""
    tool_calls_raw = delta.get("tool_calls") or []

    tool_calls = []
    for tc in tool_calls_raw:
        fn = tc.get("function") or {}
        tool_calls.append(
            AssistantPromptMessage.ToolCall(
                id=tc.get("id") or "",
                type="function",
                function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                    name=fn.get("name") or "",
                    arguments=fn.get("arguments") or "",
                ),
            )
        )

    usage_raw = chunk_data.get("usage") or {}
    usage = LLMUsage(
        prompt_tokens=usage_raw.get("prompt_tokens", 0),
        completion_tokens=usage_raw.get("completion_tokens", 0),
        total_tokens=usage_raw.get("total_tokens", 0),
        prompt_unit_price="0",
        prompt_price_unit="0",
        prompt_price="0",
        completion_unit_price="0",
        completion_price_unit="0",
        completion_price="0",
        total_price="0",
        currency="USD",
        latency=0,
    ) if usage_raw else None

    return LLMResultChunk(
        model=chunk_data.get("model", ""),
        delta=LLMResultChunkDelta(
            index=0,
            message=AssistantPromptMessage(
                content=text_content,
                tool_calls=tool_calls,
            ),
            usage=usage,
            finish_reason=finish_reason,
        ),
    )


def _openai_full_response_to_dify_result(resp_data: dict, model_name: str) -> LLMResult:
    """把 OpenAI 完整响应转成 LLMResult（非流式 fallback）"""
    choices = resp_data.get("choices") or []
    message = choices[0].get("message") or {} if choices else {}
    content = message.get("content") or ""
    tool_calls_raw = message.get("tool_calls") or []

    tool_calls = []
    for tc in tool_calls_raw:
        fn = tc.get("function") or {}
        args = fn.get("arguments") or "{}"
        tool_calls.append(
            AssistantPromptMessage.ToolCall(
                id=tc.get("id") or str(uuid.uuid4()),
                type="function",
                function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                    name=fn.get("name") or "",
                    arguments=args,
                ),
            )
        )

    usage_raw = resp_data.get("usage") or {}
    usage = LLMUsage(
        prompt_tokens=usage_raw.get("prompt_tokens", 0),
        completion_tokens=usage_raw.get("completion_tokens", 0),
        total_tokens=usage_raw.get("total_tokens", 0),
        prompt_unit_price="0",
        prompt_price_unit="0",
        prompt_price="0",
        completion_unit_price="0",
        completion_price_unit="0",
        completion_price="0",
        total_price="0",
        currency="USD",
        latency=0,
    )

    return LLMResult(
        model=model_name,
        message=AssistantPromptMessage(content=content, tool_calls=tool_calls),
        usage=usage,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Mock LLM —— 连接真实 API
# ══════════════════════════════════════════════════════════════════════════════
class MockLLM:
    """
    根据环境变量自动选择后端：
      - ANTHROPIC_API_KEY → 使用 Anthropic SDK
      - OPENAI_API_KEY    → 使用 OpenAI 兼容接口
    """

    def __init__(self):
        self._backend = self._detect_backend()
        print(f"[MockLLM] 使用后端: {self._backend}")

    def _detect_backend(self) -> str:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        raise EnvironmentError(
            "未找到 LLM API Key！\n"
            "  Anthropic: export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  OpenAI 兼容: export OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=..."
        )

    # ── Anthropic 后端 ────────────────────────────────────────────────────────
    def _call_anthropic_stream(
        self,
        model_name: str,
        prompt_messages: list[PromptMessage],
        tools: list[PromptMessageTool] | None,
    ) -> Generator[LLMResultChunk, None, None]:
        try:
            import anthropic
        except ImportError:
            raise ImportError("请先安装: pip install anthropic")

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        # 提取 system prompt
        system_text = ""
        non_system_msgs = []
        for m in prompt_messages:
            if isinstance(m, SystemPromptMessage):
                system_text += (m.content or "") + "\n"
            else:
                non_system_msgs.append(m)

        # 转换 messages
        messages = [_dify_msg_to_openai(m) for m in non_system_msgs]

        # 转换 tools
        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": 8192,
            "messages": messages,
        }
        if system_text.strip():
            kwargs["system"] = system_text.strip()
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        # 流式调用
        with client.messages.stream(**kwargs) as stream:
            tool_calls_acc: dict[int, dict] = {}
            for event in stream:
                event_type = type(event).__name__

                if event_type == "RawContentBlockStartEvent":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        idx = getattr(event, "index", 0)
                        tool_calls_acc[idx] = {
                            "id": getattr(block, "id", str(uuid.uuid4())),
                            "name": getattr(block, "name", ""),
                            "arguments": "",
                        }

                elif event_type == "RawContentBlockDeltaEvent":
                    delta = getattr(event, "delta", None)
                    if delta:
                        if getattr(delta, "type", None) == "text_delta":
                            text = getattr(delta, "text", "")
                            if text:
                                yield LLMResultChunk(
                                    model=model_name,
                                    delta=LLMResultChunkDelta(
                                        index=0,
                                        message=AssistantPromptMessage(content=text, tool_calls=[]),
                                        finish_reason=None,
                                    ),
                                )
                        elif getattr(delta, "type", None) == "input_json_delta":
                            idx = getattr(event, "index", 0)
                            if idx in tool_calls_acc:
                                tool_calls_acc[idx]["arguments"] += getattr(delta, "partial_json", "")

                elif event_type == "RawMessageStopEvent":
                    # 发送累积的 tool calls
                    if tool_calls_acc:
                        tool_calls = [
                            AssistantPromptMessage.ToolCall(
                                id=tc["id"],
                                type="function",
                                function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                    name=tc["name"],
                                    arguments=tc["arguments"],
                                ),
                            )
                            for tc in tool_calls_acc.values()
                        ]
                        yield LLMResultChunk(
                            model=model_name,
                            delta=LLMResultChunkDelta(
                                index=0,
                                message=AssistantPromptMessage(content="", tool_calls=tool_calls),
                                finish_reason="tool_calls",
                            ),
                        )

    # ── OpenAI 兼容后端 ───────────────────────────────────────────────────────
    def _call_openai_stream(
        self,
        model_name: str,
        prompt_messages: list[PromptMessage],
        tools: list[PromptMessageTool] | None,
    ) -> Generator[LLMResultChunk, None, None]:
        try:
            import openai
        except ImportError:
            raise ImportError("请先安装: pip install openai")

        base_url = os.environ.get("OPENAI_BASE_URL")
        client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=base_url if base_url else None,
        )

        messages = [_dify_msg_to_openai(m) for m in prompt_messages]
        openai_tools = _dify_tools_to_openai(tools)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        # 累积 tool_call delta（OpenAI 流式 tool calls 是分片的）
        tool_calls_acc: dict[int, dict] = {}
        last_finish_reason = None

        response = client.chat.completions.create(**kwargs)
        for raw_chunk in response:
            chunk_dict = raw_chunk.model_dump()
            choices = chunk_dict.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                last_finish_reason = finish_reason

            # 文本 delta
            text = delta.get("content") or ""
            if text:
                yield LLMResultChunk(
                    model=model_name,
                    delta=LLMResultChunkDelta(
                        index=0,
                        message=AssistantPromptMessage(content=text, tool_calls=[]),
                        finish_reason=None,
                    ),
                )

            # tool_calls delta 累积
            for tc_delta in (delta.get("tool_calls") or []):
                idx = tc_delta.get("index", 0)
                fn = tc_delta.get("function") or {}
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {"id": tc_delta.get("id") or "", "name": "", "arguments": ""}
                if tc_delta.get("id"):
                    tool_calls_acc[idx]["id"] = tc_delta["id"]
                if fn.get("name"):
                    tool_calls_acc[idx]["name"] += fn["name"]
                if fn.get("arguments"):
                    tool_calls_acc[idx]["arguments"] += fn["arguments"]

        # 流结束后发送完整 tool calls
        if tool_calls_acc:
            tool_calls = [
                AssistantPromptMessage.ToolCall(
                    id=tc["id"] or str(uuid.uuid4()),
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=tc["name"],
                        arguments=tc["arguments"],
                    ),
                )
                for tc in tool_calls_acc.values()
            ]
            yield LLMResultChunk(
                model=model_name,
                delta=LLMResultChunkDelta(
                    index=0,
                    message=AssistantPromptMessage(content="", tool_calls=tool_calls),
                    finish_reason=last_finish_reason or "tool_calls",
                ),
            )

    # ── 统一入口（对应 self.session.model.llm.invoke）────────────────────────
    def invoke(
        self,
        model_config: LLMModelConfig,
        prompt_messages: list[PromptMessage],
        tools: list[PromptMessageTool] | None = None,
        stream: bool = True,
    ) -> Generator[LLMResultChunk, None, None]:
        # 从 model_config 拿模型名，兜底用环境变量
        model_name = (
            getattr(model_config, "model", None)
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("ANTHROPIC_MODEL")
            or "claude-sonnet-4-6"
        )
        print(f"\n[MockLLM] invoke model={model_name} messages={len(prompt_messages)} tools={len(tools or [])}")

        if self._backend == "anthropic":
            return self._call_anthropic_stream(model_name, prompt_messages, tools)
        else:
            return self._call_openai_stream(model_name, prompt_messages, tools)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Mock Session
# ══════════════════════════════════════════════════════════════════════════════
class MockModelProxy:
    def __init__(self):
        self.llm = MockLLM()


class MockSession:
    def __init__(self):
        self.conversation_id = f"debug-{uuid.uuid4().hex[:8]}"
        self.storage = MockStorage()
        self.model = MockModelProxy()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Mock SkillAgentTool（替换 Dify Tool 基类）
# ══════════════════════════════════════════════════════════════════════════════
class MockSkillAgentTool:
    """
    继承 SkillAgentTool 的逻辑，但替换掉所有依赖 Dify 框架的部分：
      - self.session          → MockSession
      - self.create_text_message() → 打印到控制台并返回 ToolInvokeMessage
      - self.create_blob_message() → 仅打印路径
      - self.create_json_message() → 仅打印 JSON
    """

    def __init__(self, session: MockSession):
        self.session = session
        self.runtime_config = {}  # dify Tool 基类有时会访问

    def create_text_message(self, text: str) -> ToolInvokeMessage:
        # 打印到控制台（过滤空白）
        if text and text.strip():
            print(text, end="", flush=True)
        return ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.TEXT,
            message=ToolInvokeMessage.TextMessage(text=text),
        )

    def create_blob_message(self, blob: bytes, meta: dict | None = None) -> ToolInvokeMessage:
        print(f"[blob message: {len(blob)} bytes meta={meta}]")
        return ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.BLOB,
            message=ToolInvokeMessage.BlobMessage(blob=blob),
            meta=meta,
        )

    def create_json_message(self, obj: Any) -> ToolInvokeMessage:
        print(f"[json message]: {json.dumps(obj, ensure_ascii=False, indent=2)}")
        return ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.JSON,
            message=ToolInvokeMessage.JsonMessage(json_object=obj),
        )

    def create_link_message(self, url: str, meta: dict | None = None) -> ToolInvokeMessage:
        print(f"[link message]: {url}")
        return ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.LINK,
            message=ToolInvokeMessage.TextMessage(text=url),
        )

    # 把 SkillAgentTool._invoke 绑定过来
    def _invoke(self, tool_parameters: dict[str, Any]):
        from tools.skill_agent import SkillAgentTool
        return SkillAgentTool._invoke(self, tool_parameters)  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════════════
# 6. 构建 LLMModelConfig（填充 model_config 入参）
# ══════════════════════════════════════════════════════════════════════════════
def make_model_config(provider: str = "anthropic", model: str = "claude-sonnet-4-6") -> LLMModelConfig:
    """
    创建 LLMModelConfig，mode 固定为 chat。
    provider / model 可以随意填，因为 MockLLM.invoke 会忽略 provider，
    只用 model 字段（或兜底环境变量）。
    """
    return LLMModelConfig(
        provider=provider,
        model=model,
        model_type=ModelType.LLM,
        mode="chat",
        completion_params={},
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. 主入口
# ══════════════════════════════════════════════════════════════════════════════
def run_debug(
    query: str,
    skill_name: str = "",
    project_name: str = "",
    max_steps: int = 10,
    system_prompt: str = "你是一个通用技能执行助手，帮助用户完成各类任务。",
    model_provider: str = "anthropic",
    model_name: str = "",
):
    """
    Parameters
    ----------
    query        : 用户问题
    skill_name   : （可选）直接指定技能名称，跳过 LLM 选技能步骤
    project_name : （可选）技能所在的子项目目录名（skills/<project_name>/）
    max_steps    : 最大 LLM 步骤数
    system_prompt: 系统提示
    model_provider: 'anthropic' 或 'openai'
    model_name   : 模型名称，留空则从环境变量 OPENAI_MODEL / ANTHROPIC_MODEL 读取
    """
    # 自动推断模型名
    if not model_name:
        if model_provider == "anthropic":
            model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        else:
            model_name = os.environ.get("OPENAI_MODEL", "gpt-4o")

    print("=" * 60)
    print(f"  query      : {query}")
    print(f"  skill_name : {skill_name or '(让 LLM 自动判断)'}")
    print(f"  project    : {project_name or '(默认 skills/)'}")
    print(f"  model      : {model_provider}/{model_name}")
    print(f"  max_steps  : {max_steps}")
    print("=" * 60)
    print()

    session = MockSession()
    tool = MockSkillAgentTool(session)

    tool_parameters: dict[str, Any] = {
        "model": make_model_config(provider=model_provider, model=model_name),
        "query": query,
        "max_steps": max_steps,
        "memory_turns": 0,      # debug 时不需要历史记忆
        "history_turns": 0,
        "system_prompt": system_prompt,
    }
    if skill_name:
        tool_parameters["skill_name"] = skill_name
    if project_name:
        tool_parameters["project_name"] = project_name

    print("[输出开始] ─────────────────────────────────────────────\n")
    try:
        for msg in tool._invoke(tool_parameters):
            # 大部分输出已经在 create_text_message 里打印了
            # 这里处理 blob / json 等其他类型
            if msg.type == ToolInvokeMessage.MessageType.BLOB:
                print(f"\n[📎 Blob 文件 {len(getattr(msg.message, 'blob', b''))} bytes]")
    except KeyboardInterrupt:
        print("\n\n[中断]")
    except Exception as e:
        import traceback
        print(f"\n\n[ERROR] {e}")
        traceback.print_exc()

    print("\n[输出结束] ─────────────────────────────────────────────")


# ──────────────────────────────────────────────────────────────────────────────
# ★ 在这里修改你的测试参数 ★
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_debug(
        query="帮我搜索一下今天 Python 最新版本是多少",
        skill_name="online-search",   # 直接指定 skill，跳过自动匹配
        # project_name="project1",   # 如果 skill 在 skills/project1/ 下，取消注释
        max_steps=10,
        model_provider="anthropic",   # "anthropic" 或 "openai"
        # model_name="claude-sonnet-4-6",  # 留空则从环境变量读取
    )
