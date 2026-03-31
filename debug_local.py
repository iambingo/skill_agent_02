"""
本地 debug —— 直接跑 agent 主循环，不依赖 dify_plugin
功能与 tools/skill_agent.py 保持一致，去掉了 dify 框架、文件上传、history 存储、resume 存储。

export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://...   # 可选
export OPENAI_MODEL=gpt-4o           # 可选，默认 gpt-4o
python debug_local.py
"""

import sys, os, json, uuid, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from utils.skill_agent_runtime import _AgentRuntime
from utils.skill_agent_exec import _detect_skills_root, _cleanup_old_temp_sessions
from utils.skill_agent_schemas import TOOL_SCHEMAS, _tool_call_retry_prompt, _validate_tool_arguments
from utils.tools import _extract_first_json_object, _list_dir, _guess_mime_type, _safe_join, _shorten_text

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

temp_root = os.path.join(plugin_root, "temp")
os.makedirs(temp_root, exist_ok=True)
session_dir = os.path.join(temp_root, f"debug-{uuid.uuid4().hex[:8]}")
os.makedirs(session_dir, exist_ok=True)
_cleanup_old_temp_sessions(temp_root, keep=4, protect_dirs={session_dir})

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

# ── progressive_disclosure_rules（与原代码完全一致）─────────────────
if preselected_skill_folder:
    progressive_disclosure_rules = (
        f"【技能已预先指定】系统已为你锁定技能：《{SKILL_NAME}》（folder: {preselected_skill_folder}），其元数据已预加载。\n"
        + "你必须仅使用该技能，忽略技能索引中的其他技能，并直接按以下步骤执行：\n"
        + f"1) 调用 list_skill_files({preselected_skill_folder!r}) 查看技能包目录结构\n"
        + "2) 按需调用 read_skill_file 读取具体文件\n"
        + "3) 按说明书内容执行脚本/命令（run_skill_command）\n"
        + "4) 生成最终文件后用 export_temp_file 标记交付\n"
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
        + "7) 执行前必须先确认技能包内确实存在可执行入口（脚本/模块等），不要猜测模块名；如果缺少可执行入口，则先交付当前可交付产物，并询问用户是否允许你在 temp 目录中自行创建脚本后再尝试生成。\n"
        + "8) 按说明书要求生成最终文件后，必须用 export_temp_file 标记最终文件\n"
    )

# ── system_content（与原代码完全一致，去掉 uploads_context / resume_context）─
system_content = (
    SYSTEM_PROMPT.strip()
    + "\n\n你是一个使用 Skills 文件夹作为\u201c工具箱\u201d的通用型 Agent。\n"
    + "\n[会话路径]\n"
    + f"- session_dir: {session_dir}\n"
    + f"- skills_root: {skills_root}\n"
    + progressive_disclosure_rules
    + "路径规则：uploads/ 与你用 write_temp_file 生成的中间产物都位于 session_dir 下；run_skill_command 的 cwd 在 skills_root/<skill_name> 下。\n"
    + "因此：只要命令参数需要引用 uploads/ 或 temp 中间文件，一律使用 read_temp_file 返回的绝对路径（result.path）传给命令；不要使用 ../uploads、../../temp 这类相对路径猜测。\n"
    + "依赖安装规则：如需 npm install/npm ci/bun install，必须用 run_skill_command 在技能包内含 package.json 的目录执行（通过 cwd_relative 指到该目录）；禁止在 session_dir 执行 install，否则会写入 temp/<session>/node_modules 导致每次会话重复安装。\n"
    + "补充规则1：如果用户请求中已经明确给出具体类型/参数，则视为已确认，不要重复追问，直接进入对应分支执行。\n"
    + "补充规则2：当你需要向用户追问任何信息时：本轮必须只输出问题与选项，并立刻结束；不得在同一轮继续读取任何文件、执行任何命令、生成任何产物。\n"
    + "补充规则3：默认值只能在用户明确说'默认/随便/你决定'时启用；用户未回复不等于选择了默认。"
    + "补充规则4：当你准备调用 write_temp_file 时，必须先在自然语言里输出一行\u201c写入意图确认\u201d，包含：relative_path + 内容摘要（前 80 字）+ 大致长度；然后再发起工具调用。relative_path 必须是文件路径（不能是空、'.'、'..'、不能以 '/' 结尾，不能指向目录）。\n"
    + "你必须把实现过程中的中间产物写入 temp 会话目录（脚本、草稿、生成物等）：\n"
    + "- 写文本：write_temp_file\n"
    + "- 运行命令生成文件：run_temp_command\n"
    + "对任何\u201c有明确交付物\u201d的请求，你必须在同一轮内推进直到：生成可交付文件，或给出明确失败原因。\n"
    + "只有调用 export_temp_file 标记的文件，才会作为最终交付文件返回给用户；uploads/ 与未标记文件不会回传。\n\n"
    + "可用动作：\n"
    + "- get_session_context()\n"
    + "- get_skill_metadata(skill_name)\n"
    + "- list_skill_files(skill_name, max_depth)\n"
    + "- read_skill_file(skill_name, relative_path, max_chars)\n"
    + "- run_skill_command(skill_name, command, cwd_relative, auto_install)\n"
    + "- write_temp_file(relative_path, content)\n"
    + "- read_temp_file(relative_path, max_chars)\n"
    + "- list_temp_files(max_depth)\n"
    + "- run_temp_command(command, cwd_relative, auto_install)\n"
    + "- export_temp_file(temp_relative_path, workspace_relative_path, overwrite)  # 不复制，仅标记交付名\n\n"
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

# ── memory 压缩（与原代码一致）────────────────────────────────────
def compact() -> None:
    if MEMORY_TURNS <= 0:
        return
    keep = 1 + MEMORY_TURNS * 4
    if len(messages) > keep:
        system_msg = messages[0]
        tail = messages[-(keep - 1):]
        messages[:] = [system_msg, *tail]

# ── tool 分发（与原代码一致）─────────────────────────────────────
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
            command=arguments.get("command") if isinstance(arguments.get("command"), list) else [],
            cwd_relative=str(arguments.get("cwd_relative")) if arguments.get("cwd_relative") else None,
            auto_install=bool(arguments.get("auto_install") or False),
        )
    if name == "get_session_context":
        return runtime.get_session_context()
    if name == "write_temp_file":
        return runtime.write_temp_file(
            str(arguments.get("relative_path") or ""),
            str(arguments.get("content") or ""),
        )
    if name == "read_temp_file":
        return runtime.read_temp_file(
            str(arguments.get("relative_path") or ""),
            int(arguments.get("max_chars") or 12000),
        )
    if name == "list_temp_files":
        return runtime.list_temp_files(int(arguments.get("max_depth") or 4))
    if name == "run_temp_command":
        return runtime.run_temp_command(
            command=arguments.get("command") if isinstance(arguments.get("command"), list) else [],
            cwd_relative=str(arguments.get("cwd_relative")) if arguments.get("cwd_relative") else None,
            auto_install=bool(arguments.get("auto_install") or False),
        )
    if name == "export_temp_file":
        return runtime.export_temp_file(
            temp_relative_path=str(arguments.get("temp_relative_path") or ""),
            workspace_relative_path=str(arguments.get("workspace_relative_path") or ""),
            overwrite=bool(arguments.get("overwrite") or False),
        )
    return {"error": f"unknown tool: {name}"}

# ── gate check（与原代码一致）────────────────────────────────────
def check_gates(name: str, arguments: dict) -> str | None:
    """返回 None 表示通过；返回字符串表示拒绝原因（同时已向 messages 写入）"""
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
print(f"\nquery: {QUERY}\nskill_name: {SKILL_NAME or '(LLM 自动判断)'}\nsession_dir: {session_dir}\n{'─'*60}")

final_text: str | None = None
final_file_meta: dict[str, dict[str, str]] = {}
empty_responses = 0

for step_idx in range(MAX_STEPS):
    compact()
    print(f"\n[step {step_idx + 1}/{MAX_STEPS}]")
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOL_SCHEMAS, stream=False
    )
    msg = resp.choices[0].message
    finish = resp.choices[0].finish_reason
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

            # 参数校验
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

            # gate check
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

            # run_skill_command 失败时打印 stderr
            if name in {"run_skill_command", "run_temp_command"}:
                if isinstance(result, dict) and result.get("returncode") not in (None, 0):
                    stderr = str(result.get("stderr") or "").strip()
                    if stderr:
                        print(f"  ❌ stderr: {stderr[:800]}")

            # export_temp_file 记录交付元信息
            if name == "export_temp_file":
                temp_rel = str(arguments.get("temp_relative_path") or "")
                workspace_rel = str(arguments.get("workspace_relative_path") or "")
                out_name = os.path.basename(workspace_rel) if workspace_rel else ""
                if isinstance(result, dict) and not result.get("error") and temp_rel and out_name:
                    final_file_meta[temp_rel] = {
                        **(final_file_meta.get(temp_rel) or {}),
                        "filename": out_name,
                        "mime_type": _guess_mime_type(out_name),
                    }

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})
        continue

    # ── JSON 协议分支（模型不支持 function call 时）───────────────
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

    if name == "export_temp_file":
        temp_rel = str(arguments.get("temp_relative_path") or "")
        workspace_rel = str(arguments.get("workspace_relative_path") or "")
        out_name = os.path.basename(workspace_rel) if workspace_rel else ""
        if isinstance(result, dict) and not result.get("error") and temp_rel and out_name:
            final_file_meta[temp_rel] = {
                **(final_file_meta.get(temp_rel) or {}),
                "filename": out_name,
                "mime_type": _guess_mime_type(out_name),
            }

    messages.append({"role": "assistant", "content":
        "TOOL_RESULT\n" + json.dumps({"name": name, "result": result}, ensure_ascii=False)})

else:
    # 超出 max_steps
    try:
        has_files = any(e.get("type") == "file" for e in _list_dir(session_dir, max_depth=2) if isinstance(e, dict))
    except Exception:
        has_files = False
    if final_file_meta or has_files:
        final_text = "已生成文件。"
    else:
        final_text = f"❌超过最大执行轮数 max_steps={MAX_STEPS}，仍未得到最终结果"

# ── 输出结果 ──────────────────────────────────────────────────────
print(f"\n{'─'*60}")
if final_text and final_text.strip():
    if not final_file_meta and final_text.strip() == "已生成文件。":
        final_text = "已生成中间文件，但未调用 export_temp_file 标记交付文件。"
    print(f"\n[最终回答]\n{final_text}")

# 输出交付文件
if final_file_meta:
    print(f"\n[交付文件]")
    for rel, meta in final_file_meta.items():
        path = _safe_join(session_dir, rel.replace("\\", "/").lstrip("/"))
        out_name = meta.get("filename") or os.path.basename(rel)
        if os.path.isfile(path):
            print(f"  {out_name}  →  {path}")
        else:
            print(f"  {out_name}  ×  文件不存在: {path}")
elif not final_text:
    # 检查是否有任何 temp 文件
    try:
        has_any = any(e.get("type") == "file" for e in _list_dir(session_dir, max_depth=10) if isinstance(e, dict))
    except Exception:
        has_any = False
    if has_any:
        print("已生成中间文件，但未调用 export_temp_file 标记交付文件。")
    else:
        print("未生成任何文本或文件输出。")

print(f"\n[done]  session_dir: {session_dir}")
