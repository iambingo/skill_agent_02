"""
本地 debug —— 直接跑 agent 主循环，不依赖 dify_plugin
功能与 tools/skill_agent.py 保持一致，去掉了 dify 框架、文件上传、history 存储、resume 存储。

export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://...   # 可选
export OPENAI_MODEL=gpt-4o           # 可选，默认 gpt-4o
python debug_local.py
"""

import sys, os, json, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from utils.skill_agent_runtime import _AgentRuntime
from utils.skill_agent_exec import _detect_skills_root
from utils.skill_agent_schemas import TOOL_SCHEMAS, _tool_call_retry_prompt, _validate_tool_arguments
from utils.tools import _extract_first_json_object, _shorten_text

# ── 入参，改这里 ──────────────────────────────────────────────────
QUERY        = "帮我讲个冷笑话"
SKILL_NAME   = ""          # 留空 "" 则让 LLM 自动从 skills 列表里判断
PROJECT_NAME = ""          # 传了就用 skills/<project_name>/ 作为 skills_root
SYSTEM_PROMPT = "你是一个助手"   # 对应原代码 tool_parameters["system_prompt"]
MAX_STEPS    = 8
MEMORY_TURNS = 10
# ─────────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

plugin_root = os.path.dirname(os.path.abspath(__file__))

if PROJECT_NAME:
    _skills_base = os.environ.get("SKILLS_ROOT", "").strip() or os.path.join(plugin_root, "skills")
    skills_root = os.path.join(_skills_base, PROJECT_NAME)
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
    memory_turns=MEMORY_TURNS,
)
skills_index = runtime.load_skills_index()

# ── skill_name 预选（对应原代码 skill_name_param 逻辑）──────────────
preselected_skill_folder: str | None = None
if SKILL_NAME:
    available_skills = skills_index.get("skills") or []
    matched = next(
        (s for s in available_skills
         if str(s.get("name") or "") == SKILL_NAME or str(s.get("folder") or "") == SKILL_NAME),
        None,
    )
    if matched is None:
        names = [str(s.get("name") or s.get("folder") or "") for s in available_skills]
        print(f"❌ 指定的技能「{SKILL_NAME}」不存在。当前可用：{names}")
        sys.exit(1)
    preselected_skill_folder = str(matched.get("folder") or matched.get("name") or "").strip()
    runtime.get_skill_metadata(preselected_skill_folder)   # 预加载，满足 gate check
    skills_index = {"root": skills_index.get("root"), "skills": [matched]}
    print(f"当前选定skill为：{SKILL_NAME}")

# ── progressive_disclosure_rules ─────────────────────────────────
if preselected_skill_folder:
    progressive_disclosure_rules = (
        f"【技能已预先指定】系统已为你锁定技能：《{SKILL_NAME}》（folder: {preselected_skill_folder}），其元数据已预加载。\n"
        + "你必须仅使用该技能，忽略技能索引中的其他技能，并直接按以下步骤执行：\n"
        + f"1) 调用 list_skill_files({preselected_skill_folder!r}) 查看技能包目录结构\n"
        + "2) 按需调用 read_skill_file 读取具体文件\n"
        + "3) 按说明书内容执行脚本/命令（run_skill_command），直接输出结果\n"
    )
else:
    progressive_disclosure_rules = (
        "你必须遵循渐进式披露流程：\n"
        + "1) 只根据技能元数据（name/description）判断可能相关的技能\n"
        + "2) 触发时才调用 get_skill_metadata 读取 SKILL.md（说明文档）\n"
        + "3) 任何对技能的进一步操作（list_skill_files/read_skill_file/run_skill_command）之前，必须先 get_skill_metadata；若未执行，本系统会拒绝该调用并要求你先补读说明书。\n"
        + "4) 按说明书内容执行脚本/命令，或进一步搜索资料前，必须先调用 list_skill_files 查看技能包的目录结构，以确保在正确的目录执行命令。\n"
        + "5) 只有在需要更深信息时，才调用 read_skill_file\n"
        + "6) 只有在明确需要执行脚本/命令时，才调用 run_skill_command\n"
        + "7) 执行前必须先确认技能包内确实存在可执行入口（脚本/模块等），不要猜测模块名。\n"
        + "8) run_skill_command 执行完毕后，直接将结果输出给用户，无需额外存储步骤。\n"
    )

# ── system_content ────────────────────────────────────────────────
system_content = (
    SYSTEM_PROMPT.strip()
    + "\n\n你是一个使用 Skills 文件夹作为\u201c工具箱\u201d的通用型 Agent。\n"
    + "\n[会话路径]\n"
    + f"- skills_root: {skills_root}\n"
    + progressive_disclosure_rules
    + "依赖安装规则：如需 npm install/npm ci/bun install，必须用 run_skill_command 在技能包内含 package.json 的目录执行（通过 cwd_relative 指到该目录）。\n"
    + "补充规则1：如果用户请求中已经明确给出具体类型/参数，则视为已确认，不要重复追问，直接进入对应分支执行。\n"
    + "补充规则2：当你需要向用户追问任何信息时：本轮必须只输出问题与选项，并立刻结束；不得在同一轮继续读取任何文件、执行任何命令、生成任何产物。\n"
    + "补充规则3：默认值只能在用户明确说'默认/随便/你决定'时启用；用户未回复不等于选择了默认。\n"
    + "\n可用动作：\n"
    + "- get_skill_metadata(skill_name)\n"
    + "- list_skill_files(skill_name, max_depth)\n"
    + "- read_skill_file(skill_name, relative_path, max_chars)\n"
    + "- run_skill_command(skill_name, command, cwd_relative, auto_install)\n\n"
    + "如果模型支持 function call，请直接发起工具调用；若不支持，则用 JSON 协议响应：\n"
    + '{"type":"tool","name":"get_skill_metadata","arguments":{"skill_name":"xxx"}}\n'
    + '或 {"type":"final","content":"..."}\n\n'
    + "技能索引（用于判断是否需要调用技能）：\n"
    + json.dumps(skills_index, ensure_ascii=False)
)

messages: list[dict] = [
    {"role": "system", "content": system_content},
    {"role": "user",   "content": QUERY},
]

# ── memory 压缩 ────────────────────────────────────────────────────
def compact() -> None:
    if MEMORY_TURNS <= 0:
        return
    keep = 1 + MEMORY_TURNS * 4
    if len(messages) > keep:
        system_msg = messages[0]
        tail = messages[-(keep - 1):]
        messages[:] = [system_msg, *tail]

# ── tool 分发 ─────────────────────────────────────────────────────
def dispatch(name: str, arguments: dict):
    if name == "get_skill_metadata":
        return runtime.get_skill_metadata(str(arguments.get("skill_name") or ""))
    if name == "list_skill_files":
        return runtime.list_skill_files(
            str(arguments.get("skill_name") or ""),
            int(arguments.get("max_depth") or 2),
        )
    if name == "read_skill_file":
        return runtime.read_skill_file(
            str(arguments.get("skill_name") or ""),
            str(arguments.get("relative_path") or ""),
            int(arguments.get("max_chars") or 12000),
        )
    if name == "run_skill_command":
        return runtime.run_skill_command(
            skill_name=str(arguments.get("skill_name") or ""),
            command=arguments.get("command"),
            cwd_relative=str(arguments.get("cwd_relative")) if arguments.get("cwd_relative") else None,
            auto_install=bool(arguments.get("auto_install") or False),
        )
    return {"error": f"unknown tool: {name}"}

# ── gate check ────────────────────────────────────────────────────
def check_gates(name: str, arguments: dict) -> str | None:
    if name in {"list_skill_files", "read_skill_file", "run_skill_command"}:
        skill_name = str(arguments.get("skill_name") or "").strip()
        if skill_name and not runtime.has_skill_metadata(skill_name):
            result = {
                "error": "skill_md_required",
                "skill_name": skill_name,
                "detail": "必须先调用 get_skill_metadata(skill_name) 读取 SKILL.md（说明书）后，才能继续调用该工具。",
            }
            return json.dumps(result, ensure_ascii=False)
        if name == "run_skill_command" and skill_name and not runtime.has_listed_skill_files(skill_name):
            result = {
                "error": "skill_files_listing_required",
                "skill_name": skill_name,
                "detail": "执行技能命令前，必须先调用 list_skill_files(skill_name) 查看技能包目录结构。",
            }
            return json.dumps(result, ensure_ascii=False)
    return None

# ── 主循环 ────────────────────────────────────────────────────────
print(f"\nquery: {QUERY}\nskill_name: {SKILL_NAME or '(LLM 自动判断)'}\n{'─'*60}")

final_text: str | None = None
empty_responses = 0

for step_idx in range(MAX_STEPS):
    compact()
    print(f"\n[step {step_idx + 1}/{MAX_STEPS}]")
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOL_SCHEMAS, stream=False
    )
    msg = resp.choices[0].message
    res_text = msg.content or ""
    tool_calls = msg.tool_calls or []

    if res_text:
        print(res_text)

    # ── function call 分支 ────────────────────────────────────────
    if tool_calls:
        empty_responses = 0
        messages.append(msg)
        for tc in tool_calls:
            name = tc.function.name
            arguments = json.loads(tc.function.arguments or "{}")
            print(f"  → {name}({_shorten_text(arguments, 200)})")

            ok_args, arg_detail = _validate_tool_arguments(name, arguments)
            if not ok_args:
                result_str = json.dumps(
                    {"error": "invalid_tool_arguments", "tool": name, "detail": arg_detail, "got": arguments},
                    ensure_ascii=False,
                )
                print(f"  ✗ invalid args: {arg_detail}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
                messages.append({"role": "user", "content": _tool_call_retry_prompt(name, arg_detail)})
                continue

            gate_err = check_gates(name, arguments)
            if gate_err:
                gate_obj = json.loads(gate_err)
                skill_name = str(arguments.get("skill_name") or "")
                print(f"  ✗ gate: {gate_obj.get('error')}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": gate_err})
                if gate_obj.get("error") == "skill_md_required":
                    messages.append({"role": "user", "content":
                        f"你刚才尝试调用 `{name}` 但尚未读取技能《{skill_name}》的 SKILL.md。"
                        f"请先调用 get_skill_metadata({skill_name!r})，再重试该工具调用。"})
                else:
                    messages.append({"role": "user", "content":
                        f"你刚才尝试调用 `{name}` 但尚未查看技能《{skill_name}》的目录结构。"
                        f"请先调用 list_skill_files({skill_name!r})，再重试该工具调用。"})
                continue

            result = dispatch(name, arguments)
            print(f"  ← {json.dumps(result, ensure_ascii=False)[:300]}")

            if name == "run_skill_command":
                if isinstance(result, dict) and result.get("returncode") not in (None, 0):
                    stderr = str(result.get("stderr") or "").strip()
                    if stderr:
                        print(f"  ❌ stderr: {stderr[:800]}")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})
        continue

    # ── JSON 协议分支 ─────────────────────────────────────────────
    json_text = _extract_first_json_object(res_text)
    action: dict | None = None
    if json_text:
        try:
            action = json.loads(json_text)
        except Exception:
            action = None

    if not res_text and not action:
        empty_responses += 1
        if empty_responses < 3:
            messages.append({"role": "user", "content":
                '你刚才没有输出任何内容。请继续完成任务：如果支持函数调用请调用工具；否则请输出 JSON：{"type":"final","content":"..."}'})
            continue
        final_text = "模型连续返回空响应，未生成任何结果。"
        break

    if not action or action.get("type") == "final":
        final_text = str(action.get("content") or "") if action else res_text
        break

    if action.get("type") != "tool":
        final_text = res_text
        break

    name = str(action.get("name") or "")
    arguments = action.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}

    print(f"  → [JSON] {name}({_shorten_text(arguments, 200)})")

    ok_args, arg_detail = _validate_tool_arguments(name, arguments)
    if not ok_args:
        messages.append({"role": "user", "content": _tool_call_retry_prompt(name, arg_detail)})
        messages.append({"role": "assistant", "content":
            "TOOL_RESULT\n" + json.dumps({"name": name, "result": {"error": "invalid_tool_arguments", "detail": arg_detail}}, ensure_ascii=False)})
        continue

    gate_err = check_gates(name, arguments)
    if gate_err:
        gate_obj = json.loads(gate_err)
        skill_name = str(arguments.get("skill_name") or "")
        if gate_obj.get("error") == "skill_md_required":
            messages.append({"role": "user", "content":
                f"你刚才尝试调用 `{name}` 但尚未读取技能《{skill_name}》的 SKILL.md。"
                f"请先调用 get_skill_metadata({skill_name!r})，再重试该工具调用。"})
        else:
            messages.append({"role": "user", "content":
                f"你刚才尝试调用 `{name}` 但尚未查看技能《{skill_name}》的目录结构。"
                f"请先调用 list_skill_files({skill_name!r})，再重试该工具调用。"})
        messages.append({"role": "assistant", "content": "TOOL_RESULT\n" + gate_err})
        continue

    messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
    result = dispatch(name, arguments)
    print(f"  ← {json.dumps(result, ensure_ascii=False)[:300]}")

    if name == "run_skill_command":
        if isinstance(result, dict) and result.get("returncode") not in (None, 0):
            stderr = str(result.get("stderr") or "").strip()
            if stderr:
                print(f"  ❌ stderr: {stderr[:800]}")

    messages.append({"role": "assistant", "content":
        "TOOL_RESULT\n" + json.dumps({"name": name, "result": result}, ensure_ascii=False)})

else:
    final_text = f"❌超过最大执行轮数 max_steps={MAX_STEPS}，仍未得到最终结果"

# ── 输出结果 ──────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"\n[最终回答]\n{final_text or '未生成任何文本输出。'}")
print(f"\n[done]")
