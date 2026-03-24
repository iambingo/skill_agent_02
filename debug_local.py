"""
本地 debug —— 直接跑 agent 主循环，不依赖 dify_plugin

export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://...   # 可选
export OPENAI_MODEL=gpt-4o           # 可选，默认 gpt-4o
python debug_local.py
"""

import sys, os, json, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from utils.skill_agent_runtime import _AgentRuntime
from utils.skill_agent_exec import _detect_skills_root
from utils.skill_agent_schemas import TOOL_SCHEMAS

# ── 入参，改这里 ──────────────────────────────────────────────────
QUERY        = "帮我搜索一下今天 Python 最新版本是多少"
SKILL_NAME   = "online-search"   # 留空 "" 则让 LLM 自动从 skills 列表里判断
PROJECT_NAME = ""                # 传了就用 skills/<project_name>/ 作为 skills_root
MAX_STEPS    = 10
# ─────────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

plugin_root = os.path.dirname(os.path.abspath(__file__))

if PROJECT_NAME:
    skills_root = os.path.join(plugin_root, "skills", PROJECT_NAME)
    if not os.path.isdir(skills_root):
        print(f"❌ 项目「{PROJECT_NAME}」不存在，路径: {skills_root}")
        sys.exit(1)
else:
    skills_root = _detect_skills_root(None)
session_dir = os.path.join(plugin_root, "temp", f"debug-{uuid.uuid4().hex[:8]}")
os.makedirs(session_dir, exist_ok=True)

runtime = _AgentRuntime(
    skills_root=skills_root,
    session_dir=session_dir,
    max_steps=MAX_STEPS,
    memory_turns=0,
)
skills_index = runtime.load_skills_index()

# ── system prompt ─────────────────────────────────────────────────
if SKILL_NAME:
    # 找到匹配的 skill，只暴露这一个
    matched = next((s for s in skills_index["skills"]
                    if s["name"] == SKILL_NAME or s["folder"] == SKILL_NAME), None)
    if not matched:
        print(f"❌ skill 「{SKILL_NAME}」不存在，当前有：{[s['folder'] for s in skills_index['skills']]}")
        sys.exit(1)
    skills_index = {"root": skills_index["root"], "skills": [matched]}
    runtime.get_skill_metadata(matched["folder"])   # 预加载 SKILL.md
    rules = (
        f"技能已锁定：《{SKILL_NAME}》（folder: {matched['folder']}），SKILL.md 已预加载。\n"
        "直接按以下步骤：1) list_skill_files 2) run_skill_command 3) export_temp_file"
    )
else:
    rules = (
        "渐进式披露：先看 skills_index 选技能 → get_skill_metadata → list_skill_files → run_skill_command"
    )

system_prompt = f"""你是一个通用技能执行助手。
session_dir: {session_dir}
skills_root: {skills_root}
{rules}
技能索引：
{json.dumps(skills_index, ensure_ascii=False)}"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": QUERY},
]

# ── tool 分发 ─────────────────────────────────────────────────────
def dispatch(name: str, args: dict):
    if name == "get_skill_metadata":
        return runtime.get_skill_metadata(args["skill_name"])
    if name == "list_skill_files":
        return runtime.list_skill_files(args["skill_name"], int(args.get("max_depth", 2)))
    if name == "read_skill_file":
        return runtime.read_skill_file(args["skill_name"], args["relative_path"], int(args.get("max_chars", 12000)))
    if name == "run_skill_command":
        return runtime.run_skill_command(
            skill_name=args["skill_name"],
            command=args.get("command", []),
            cwd_relative=args.get("cwd_relative"),
            auto_install=bool(args.get("auto_install", False)),
        )
    if name == "write_temp_file":
        return runtime.write_temp_file(args["relative_path"], args["content"])
    if name == "read_temp_file":
        return runtime.read_temp_file(args["relative_path"], int(args.get("max_chars", 12000)))
    if name == "list_temp_files":
        return runtime.list_temp_files(int(args.get("max_depth", 4)))
    if name == "run_temp_command":
        return runtime.run_temp_command(command=args.get("command", []),
                                         cwd_relative=args.get("cwd_relative"),
                                         auto_install=bool(args.get("auto_install", False)))
    if name == "export_temp_file":
        return runtime.export_temp_file(args["temp_relative_path"],
                                         args.get("workspace_relative_path", ""),
                                         bool(args.get("overwrite", False)))
    if name == "get_session_context":
        return runtime.get_session_context()
    return {"error": f"unknown tool: {name}"}

# ── 主循环 ────────────────────────────────────────────────────────
print(f"\nquery: {QUERY}\nskill_name: {SKILL_NAME or '(LLM 自动判断)'}\n{'─'*50}")

for step in range(MAX_STEPS):
    print(f"\n[step {step+1}]")
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOL_SCHEMAS, stream=False
    )
    msg = resp.choices[0].message
    finish = resp.choices[0].finish_reason

    if msg.content:
        print(msg.content)

    messages.append(msg)   # openai 对象可直接 append

    if finish == "stop" or not msg.tool_calls:
        print("\n[done]")
        break

    for tc in msg.tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments or "{}")
        print(f"  → {name}({args})")
        result = dispatch(name, args)
        print(f"  ← {json.dumps(result, ensure_ascii=False)[:300]}")
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result, ensure_ascii=False),
        })
