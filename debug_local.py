"""
本地 debug 脚本 —— 完全绕开 dify_plugin，用 openai SDK 驱动 SkillAgentTool

用法：
    export OPENAI_API_KEY=sk-xxx
    export OPENAI_BASE_URL=https://...   # 可选，默认 OpenAI
    export OPENAI_MODEL=gpt-4o          # 可选，默认 gpt-4o
    python debug_local.py
"""

import sys, os, json, uuid, types

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────
# 1. 在 sys.modules 里注入假的 dify_plugin，让 skill_agent.py 能正常 import
# ─────────────────────────────────────────────────────────────────

class _Msg:
    """通用消息基类，_safe_get 通过 getattr 访问字段"""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

class _Role:
    SYSTEM = "system"; USER = "user"; ASSISTANT = "assistant"; TOOL = "tool"

class _ToolCallFunc(_Msg): pass  # .name .arguments(str)
class _ToolCall(_Msg):           # .id .type .function(_ToolCallFunc)
    ToolCallFunction = _ToolCallFunc

class _SystemMsg(_Msg):
    role = _Role.SYSTEM
class _UserMsg(_Msg):
    role = _Role.USER
class _AssistantMsg(_Msg):
    role = _Role.ASSISTANT
    ToolCall = _ToolCall
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
class _ToolMsg(_Msg):
    role = _Role.TOOL

class _PromptTool(_Msg): pass     # .name .description .parameters

class _InvokeMsg:
    class MessageType:
        TEXT = "text"; BLOB = "blob"; JSON = "json"; LINK = "link"
    class TextMessage(_Msg): pass
    class BlobMessage(_Msg): pass
    class JsonMessage(_Msg): pass
    def __init__(self, type, message, meta=None):
        self.type = type; self.message = message; self.meta = meta

class _Tool:
    """Tool 基类占位，实际方法由 MockTool 覆盖"""
    pass

# 构造伪 dify_plugin 包树
def _make_mod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

_dp          = _make_mod("dify_plugin")
_dp_msg      = _make_mod("dify_plugin.entities")
_dp_msg2     = _make_mod("dify_plugin.entities.model")
_dp_msg3     = _make_mod("dify_plugin.entities.model.message")
_dp_tool_mod = _make_mod("dify_plugin.entities.tool")

_dp.Tool = _Tool
_dp_msg3.SystemPromptMessage  = _SystemMsg
_dp_msg3.UserPromptMessage    = _UserMsg
_dp_msg3.AssistantPromptMessage = _AssistantMsg
_dp_msg3.ToolPromptMessage    = _ToolMsg
_dp_msg3.PromptMessageTool    = _PromptTool
_dp_tool_mod.ToolInvokeMessage = _InvokeMsg

# ─────────────────────────────────────────────────────────────────
# 2. Mock Storage
# ─────────────────────────────────────────────────────────────────
class MockStorage:
    def __init__(self): self._d = {}
    def get(self, key): return self._d.get(key)
    def set(self, key, val): self._d[key] = val
    def delete(self, key): self._d.pop(key, None)

# ─────────────────────────────────────────────────────────────────
# 3. Mock LLM —— 用 openai SDK 调真实接口，返回 skill_agent 能读的对象
# ─────────────────────────────────────────────────────────────────
class MockLLM:
    def invoke(self, model_config, prompt_messages, tools=None, stream=True):
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")

        # 转 openai messages 格式
        msgs = []
        for m in prompt_messages:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "") or ""
            if role == _Role.TOOL:
                msgs.append({"role": "tool", "tool_call_id": m.tool_call_id,
                             "content": content})
            elif role == _Role.ASSISTANT:
                d = {"role": "assistant", "content": content}
                if getattr(m, "tool_calls", None):
                    d["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in m.tool_calls
                    ]
                msgs.append(d)
            else:
                msgs.append({"role": role, "content": content})

        # 转 openai tools 格式
        oai_tools = None
        if tools:
            oai_tools = [{"type": "function", "function": {
                "name": t.name, "description": t.description,
                "parameters": t.parameters}} for t in tools]

        kw = dict(model=model, messages=msgs, stream=True)
        if oai_tools:
            kw["tools"] = oai_tools

        # 流式响应 —— yield 简单对象让 skill_agent 的 _safe_get 能读
        return self._stream(client, kw)

    @staticmethod
    def _stream(client, kw):
        tc_acc = {}   # index → {id, name, arguments}
        for chunk in client.chat.completions.create(**kw):
            c = chunk.choices[0] if chunk.choices else None
            if not c:
                continue
            delta = c.delta
            text  = getattr(delta, "content", "") or ""
            tcs   = getattr(delta, "tool_calls", None) or []

            for tc in tcs:
                i = tc.index
                if i not in tc_acc:
                    tc_acc[i] = {"id": tc.id or "", "name": "", "arguments": ""}
                if tc.id:
                    tc_acc[i]["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn:
                    tc_acc[i]["name"]      += getattr(fn, "name", "") or ""
                    tc_acc[i]["arguments"] += getattr(fn, "arguments", "") or ""

            finish = c.finish_reason

            # 构造 skill_agent 期望的 chunk 结构
            tool_call_objs = []
            if finish in ("tool_calls", "stop") and tc_acc:
                for info in tc_acc.values():
                    tool_call_objs.append(_ToolCall(
                        id=info["id"], type="function",
                        function=_ToolCallFunc(name=info["name"],
                                               arguments=info["arguments"])))
                tc_acc.clear()

            msg = _AssistantMsg(content=text, tool_calls=tool_call_objs)
            delta_obj = _Msg(message=msg)
            yield _Msg(delta=delta_obj)

# ─────────────────────────────────────────────────────────────────
# 4. Mock Session
# ─────────────────────────────────────────────────────────────────
class MockSession:
    def __init__(self):
        self.conversation_id = f"debug-{uuid.uuid4().hex[:8]}"
        self.storage = MockStorage()
        self.model = _Msg(llm=MockLLM())

# ─────────────────────────────────────────────────────────────────
# 5. MockTool —— 替换 Tool 基类的输出方法
# ─────────────────────────────────────────────────────────────────
class MockTool:
    def __init__(self, session):
        self.session = session

    def create_text_message(self, text):
        if text and text.strip():
            print(text, end="", flush=True)
        return _InvokeMsg(_InvokeMsg.MessageType.TEXT, _InvokeMsg.TextMessage(text=text))

    def create_blob_message(self, blob, meta=None):
        print(f"[blob {len(blob)} bytes]")
        return _InvokeMsg(_InvokeMsg.MessageType.BLOB, _InvokeMsg.BlobMessage(blob=blob), meta)

    def create_json_message(self, obj):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        return _InvokeMsg(_InvokeMsg.MessageType.JSON, _InvokeMsg.JsonMessage(json_object=obj))

    def create_link_message(self, url, meta=None):
        print(f"[link] {url}")
        return _InvokeMsg(_InvokeMsg.MessageType.LINK, _InvokeMsg.TextMessage(text=url))

    def _invoke(self, params):
        from tools.skill_agent import SkillAgentTool
        return SkillAgentTool._invoke(self, params)  # type: ignore

# ─────────────────────────────────────────────────────────────────
# 6. 运行
# ─────────────────────────────────────────────────────────────────
def run(query, skill_name="", project_name="", max_steps=10):
    tool = MockTool(MockSession())
    params = {
        "model": object(),            # MockLLM.invoke 不用这个值
        "query": query,
        "max_steps": max_steps,
        "memory_turns": 0,
        "history_turns": 0,
        "system_prompt": "你是一个通用技能执行助手。",
    }
    if skill_name:
        params["skill_name"] = skill_name
    if project_name:
        params["project_name"] = project_name

    print(f"\n{'='*50}\nquery: {query}\nskill_name: {skill_name or '(LLM 自动判断)'}\n{'='*50}\n")
    for _ in tool._invoke(params):
        pass
    print("\n[done]")


if __name__ == "__main__":
    run(
        query="帮我搜索一下今天 Python 最新版本是多少",
        skill_name="online-search",   # 留空则让 LLM 自动选
        # project_name="project1",
    )
