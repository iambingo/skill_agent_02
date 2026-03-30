import re
import json
import os
import uuid
from collections.abc import Generator
from typing import Any

from utils.tools import (
    _build_prompt_message_tools,
    _download_file_content,
    _extract_first_json_object,
    _extract_url_and_name,
    _guess_mime_type,
    _infer_ext_from_url,
    _parse_tool_call,
    _safe_filename,
    _safe_get,
    _safe_join,
    _shorten_text,
    _split_message_content,
 )

from utils.skill_agent_constants import HISTORY_TRANSCRIPT_MAX_CHARS
from utils.skill_agent_debug import _dbg, _model_brief
from utils.skill_agent_exec import _cleanup_old_temp_sessions, _detect_skills_root
from utils.skill_agent_runtime import _AgentRuntime
from utils.skill_agent_schemas import TOOL_SCHEMAS, _tool_call_retry_prompt, _validate_tool_arguments
from utils.skill_agent_storage import (
    _append_history_turn,
    _get_history_storage_key,
    _storage_get_json,
    _storage_set_json,
)
from utils.skill_agent_uploads import _build_uploads_context

from dify_plugin import Tool
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.tool import ToolInvokeMessage

class SkillAgentTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        model = tool_parameters.get("model")
        query = tool_parameters.get("query")
        max_steps = int(tool_parameters.get("max_steps") or 8)
        memory_turns = int(tool_parameters.get("memory_turns") or 10)
        history_turns = int(tool_parameters.get("history_turns") or 0)
        system_prompt = tool_parameters.get("system_prompt") or "你是一个xxxx"
        project_name = str(tool_parameters.get("project_name") or "").strip()
        if project_name:
            _plugin_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            _project_dir = os.path.join(_plugin_root, "skills", project_name)
            if not os.path.isdir(_project_dir):
                yield self.create_text_message(
                    f"❌ 项目「{project_name}」不存在，请先通过技能管理工具创建该项目。\n"
                )
                return
            skills_root = _project_dir
        else:
            skills_root = _detect_skills_root(tool_parameters.get("skills_root"))

        if not query or not isinstance(query, str):
            yield self.create_text_message("❌缺少 query 参数\n")
            return
        user_input = str(query)

        storage = self.session.storage
        history_key = _get_history_storage_key(self.session)

        plugin_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        temp_root = os.path.join(plugin_root, "temp")
        os.makedirs(temp_root, exist_ok=True)
        session_dir = os.path.join(temp_root, f"dify-skill-{uuid.uuid4().hex[:8]}")
        os.makedirs(session_dir, exist_ok=True)
        _cleanup_old_temp_sessions(temp_root, keep=4, protect_dirs={session_dir})

        file_items: list[Any] = []
        files_param = tool_parameters.get("files")
        if isinstance(files_param, list):
            file_items = [x for x in files_param if x]
        elif files_param:
            file_items = [files_param]
        elif tool_parameters.get("file"):
            file_items = [tool_parameters.get("file")]

        uploads_context = ""
        if file_items:
            uploads_dir = _safe_join(session_dir, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            uploaded: list[dict[str, Any]] = []
            for item in file_items:
                url, name = _extract_url_and_name(item)
                if not url:
                    yield self.create_text_message("❌未能获取上传文件 URL（files[i].url）。\n")
                    return
                try:
                    content = _download_file_content(str(url), timeout=45)
                except Exception as e:
                    yield self.create_text_message(f"❌文件下载失败：{str(e)}\n")
                    return
                ext = _infer_ext_from_url(str(url))
                filename = _safe_filename(str(name) if name else None, fallback_ext=ext)
                abs_path = os.path.join(uploads_dir, filename)
                try:
                    with open(abs_path, "wb") as f:
                        f.write(content)
                except Exception as e:
                    yield self.create_text_message(f"❌保存上传文件失败：{str(e)}\n")
                    return

                rel_path = f"uploads/{filename}"
                mime = None
                if isinstance(item, dict) and item.get("mime_type"):
                    mime = str(item.get("mime_type") or "").strip() or None
                if not mime:
                    try:
                        mime = _guess_mime_type(filename)
                    except Exception:
                        mime = None
                uploaded.append(
                    {
                        "relative_path": rel_path,
                        "bytes": len(content),
                        "mime_type": mime or "",
                        "filename": filename,
                        "source_url": str(url),
                    }
                )

            lines = ["\n\n[上传文件清单]", "以下路径均相对于本次会话的 session_dir："]
            for f in uploaded:
                lines.append(
                    f"- {f.get('relative_path')} | mime={f.get('mime_type') or ''} | bytes={f.get('bytes') or 0} | filename={f.get('filename') or ''}"
                )
            uploads_context = "\n".join(lines) + "\n"
        else:
            uploads_dir = _safe_join(session_dir, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)

        uploads_context = _build_uploads_context(session_dir)

        runtime = _AgentRuntime(
            skills_root=skills_root,
            session_dir=session_dir,
            max_steps=max_steps,
            memory_turns=memory_turns,
        )

        history_messages: list[Any] = []
        if history_turns > 0:
            history_state = _storage_get_json(storage, history_key)
            turns = history_state.get("turns")
            if isinstance(turns, list) and turns:
                picked: list[tuple[str, str]] = []
                for t in reversed(turns[-history_turns:]):
                    if not isinstance(t, dict):
                        continue
                    u = str(t.get("user") or "").strip()
                    a = str(t.get("assistant") or "").strip()
                    if not u and not a:
                        continue
                    picked.append((u, a))
                if picked:
                    acc: list[tuple[str, str]] = []
                    total = 0
                    for u, a in picked:
                        block_len = len(u) + len(a)
                        if total + block_len > HISTORY_TRANSCRIPT_MAX_CHARS and acc:
                            break
                        acc.append((u, a))
                        total += block_len
                        if total >= HISTORY_TRANSCRIPT_MAX_CHARS:
                            break
                    acc.reverse()
                    for u, a in acc:
                        if u:
                            history_messages.append(UserPromptMessage(content=u))
                        if a:
                            history_messages.append(AssistantPromptMessage(content=a))

        skills_index = runtime.load_skills_index()
        try:
            skills_count = len(skills_index.get("skills") or []) if isinstance(skills_index, dict) else 0
        except Exception:
            skills_count = 0
        # Handle optional skill_name parameter (bypass LLM skill selection)
        skill_name_param = str(tool_parameters.get("skill_name") or "").strip()
        preselected_skill_folder: str | None = None
        if skill_name_param:
            available_skills: list[dict[str, Any]] = (
                skills_index.get("skills") or [] if isinstance(skills_index, dict) else []
            )
            matched: dict[str, Any] | None = None
            for s in available_skills:
                if (
                    str(s.get("name") or "") == skill_name_param
                    or str(s.get("folder") or "") == skill_name_param
                ):
                    matched = s
                    break
            if matched is None:
                available_names = [
                    str(s.get("name") or s.get("folder") or "") for s in available_skills
                ]
                names_str = ", ".join(available_names) if available_names else "(无)"
                yield self.create_text_message(
                    f"❌ 指定的技能「{skill_name_param}」在当前项目中不存在。\n"
                    f"当前可用技能：{names_str}\n"
                )
                return
            preselected_skill_folder = str(
                matched.get("folder") or matched.get("name") or ""
            ).strip()
            # Pre-load metadata to satisfy progressive disclosure gate checks
            runtime.get_skill_metadata(preselected_skill_folder)
            # Filter skills index to only expose the preselected skill
            skills_index = {"root": skills_index.get("root"), "skills": [matched]}
            yield self.create_text_message(f"当前选定skill为：{skill_name_param}\n")

        _dbg(
            "start "
            + _model_brief(model)
            + f" session_dir={session_dir} skills_root={skills_root!s} skills_count={skills_count} "
            + f"query_len={len(query)}"
        )
        if preselected_skill_folder:
            progressive_disclosure_rules = (
                f"【技能已预先指定】系统已为你锁定技能：《{skill_name_param}》（folder: {preselected_skill_folder}），其元数据已预加载。\n"
                + "你必须仅使用该技能，忽略技能索引中的其他技能，并直接按以下步骤执行：\n"
                + f"1) 调用 list_skill_files({preselected_skill_folder!r}) 查看技能包目录结构\n"
                + "2) 按需调用 read_skill_file 读取具体文件（包括 SKILL.md 了解参数）\n"
                + "3) 按说明书内容执行脚本/命令（run_skill_command）\n"
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
            )

        system_content = (
            system_prompt.strip()
            + "\n\n你是一个使用 Skills 文件夹作为“工具箱”的通用型 Agent。\n"
            + "\n[会话路径]\n"
            + f"- skills_root: {skills_root}\n"
            + progressive_disclosure_rules
            + "依赖安装规则：如需 npm install/npm ci/bun install，必须用 run_skill_command 在技能包内含 package.json 的目录执行（通过 cwd_relative 指到该目录）。\n"
            + "补充规则1：如果用户请求中已经明确给出具体类型/参数，则视为已确认，不要重复追问，直接进入对应分支执行。\n"
            + "补充规则2：当你需要向用户追问任何信息时：本轮必须只输出问题与选项，并立刻结束；不得在同一轮继续读取任何文件、执行任何命令。\n"
            + "补充规则3：默认值只能在用户明确说'默认/随便/你决定'时启用；用户未回复不等于选择了默认。\n"
            + (uploads_context or "")
            + "可用动作：\n"
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

        messages: list[Any] = [SystemPromptMessage(content=system_content)]
        if history_messages:
            messages.extend(history_messages)
        messages.append(UserPromptMessage(content=query))

        def compact() -> None:
            if memory_turns <= 0:
                return
            keep = 1 + memory_turns * 4
            if len(messages) > keep:
                system_msg = messages[0]
                tail = messages[-(keep - 1) :]
                messages[:] = [system_msg, *tail]

        final_text: str | None = None
        empty_responses = 0
        final_text_already_streamed = False

        def stream_text_to_user(text: str, chunk_size: int = 8) -> Generator[ToolInvokeMessage]:
            s = (text or "").strip()
            if not s:
                return
            step = max(1, int(chunk_size))
            for i in range(0, len(s), step):
                yield self.create_text_message(s[i : i + step])

        def redact_user_visible_text(text: str) -> str:
            s = str(text or "")
            if not s:
                return s
            for p in [session_dir, skills_root]:
                if p and isinstance(p, str):
                    s = s.replace(p, "<REDACTED_PATH>")
                    s = s.replace(p.replace("\\", "/"), "<REDACTED_PATH>")
            s = re.sub(r"[A-Za-z]:\\[^\s\r\n\t\"']+", "<REDACTED_PATH>", s)
            s = re.sub(r"/[^\s\r\n\t\"']+", "<REDACTED_PATH>", s)
            return s

        def invoke_llm_live(
            *, prompt_messages: list[Any], tools: list[Any] | None
        ) -> Generator[ToolInvokeMessage, None, tuple[str, list[Any], Any, int, bool]]:
            nontext_content: list[dict[str, Any]] = []
            tool_calls_all: list[Any] = []
            text_parts: list[str] = []
            chunks_count = 0
            streamed_any = False
            saw_tool_calls = False
            typing_chunk = 6
            emitted_prefix = False
            emitted_len = 0

            def emit_typing(text: str) -> Generator[ToolInvokeMessage, None, None]:
                nonlocal streamed_any
                if not text:
                    return
                tagged = text.strip() + "\n\n"
                step = max(1, int(typing_chunk))
                for i in range(0, len(tagged), step):
                    yield self.create_text_message(tagged[i : i + step])
                    streamed_any = True
            
            def should_emit_user_text(text: str) -> bool:
                if not text:
                    return False
                s = str(text)
                stripped = s.lstrip()
                if stripped.startswith("{") and _extract_first_json_object(s) is None:
                    return False
                if stripped.startswith("```") and stripped.count("```") < 2:
                    return False
                json_text = _extract_first_json_object(text)
                if not json_text:
                    return True
                try:
                    obj = json.loads(json_text)
                except Exception:
                    return True
                if not isinstance(obj, dict):
                    return True
                t = obj.get("type")
                return t not in {"tool", "final"}

            try:
                try:
                    response = self.session.model.llm.invoke(
                        model_config=model,
                        prompt_messages=prompt_messages,
                        tools=tools,
                        stream=True,
                    )
                except TypeError:
                    response = self.session.model.llm.invoke(
                        model_config=model,
                        prompt_messages=prompt_messages,
                        stream=True,
                    )

                if _safe_get(response, "message") is not None:
                    msg = _safe_get(response, "message") or {}
                    content = _safe_get(msg, "content")
                    text, parts = _split_message_content(content)
                    if parts:
                        nontext_content.extend(parts)
                    tool_calls = _safe_get(msg, "tool_calls") or []
                    if isinstance(tool_calls, list):
                        tool_calls_all.extend(tool_calls)
                        if tool_calls:
                            saw_tool_calls = True
                    if text:
                        text_parts.append(text)
                    combined_text = "".join(text_parts).strip()
                    if combined_text and not saw_tool_calls and should_emit_user_text(combined_text):
                        yield from emit_typing(combined_text)
                    return combined_text, tool_calls_all, nontext_content, chunks_count, streamed_any

                for chunk in response:
                    chunks_count += 1
                    delta = _safe_get(chunk, "delta") or {}
                    msg = _safe_get(delta, "message") or {}
                    content = _safe_get(msg, "content")
                    t, parts = _split_message_content(content)
                    if parts:
                        nontext_content.extend(parts)
                    tc = _safe_get(msg, "tool_calls") or []
                    if isinstance(tc, list) and tc:
                        tool_calls_all.extend(tc)
                        if not saw_tool_calls:
                            saw_tool_calls = True
                    if t:
                        text_parts.append(t)
                        combined_text_live = "".join(text_parts).strip()
                        if combined_text_live and not saw_tool_calls and should_emit_user_text(combined_text_live):
                            if not emitted_prefix:
                                emitted_prefix = True
                            new = combined_text_live[emitted_len:]
                            if new:
                                step = max(1, int(typing_chunk))
                                for i in range(0, len(new), step):
                                    yield self.create_text_message(new[i : i + step])
                                    streamed_any = True
                                emitted_len = len(combined_text_live)
                combined_text = "".join(text_parts).strip()
                if emitted_prefix:
                    yield self.create_text_message("\n\n")
                elif combined_text and not saw_tool_calls and should_emit_user_text(combined_text):
                    yield from emit_typing(combined_text)
                return combined_text, tool_calls_all, nontext_content, chunks_count, streamed_any
            except Exception as e:
                return "", [], {"error": "stream_parse_failed", "exception": str(e)}, chunks_count, streamed_any

        try:
            for step_idx in range(max_steps):
                compact()
                _dbg(f"step={step_idx+1}/{max_steps} messages={len(messages)}")
                try:
                    res_text, tool_calls, nontext, chunks, streamed_any = yield from invoke_llm_live(
                        prompt_messages=messages,
                        tools=_build_prompt_message_tools(TOOL_SCHEMAS, PromptMessageTool),
                    )
                except Exception as e:
                    msg = str(e)
                    if "NameResolutionError" in msg or "Failed to resolve" in msg:
                        yield self.create_text_message(
                            "❌ LLM 调用失败：无法解析模型服务域名（DNS/网络问题）。\n"
                            "当前报错信息：\n"
                            + msg
                            + "\n\n请检查：\n"
                            + "1) 运行插件的环境是否能访问公网/是否需要代理\n"
                            + "2) DNS 是否可用（能否解析 dashscope.aliyuncs.com 等域名）\n"
                            + "3) Dify 的模型供应商（通义）网络出站是否被限制\n"
                        )
                    else:
                        yield self.create_text_message("❌ LLM 调用失败：\n" + msg)
                    return

                _dbg(
                    f"llm_return content_len={len(res_text)} tool_calls={len(tool_calls)} chunks={chunks} "
                    f"nontext={_shorten_text(nontext, 200) if nontext else ''}"
                )
                if tool_calls:
                    empty_responses = 0
                    messages.append(AssistantPromptMessage(content=res_text or "", tool_calls=tool_calls))
                    for tc in tool_calls:
                        call_id, name, arguments = _parse_tool_call(tc)
                        tool_name = str(name or "")
                        _dbg(f"tool_call name={tool_name} id={call_id!s} args={_shorten_text(arguments, 400)}")

                        ok_args, arg_detail = _validate_tool_arguments(tool_name, arguments)
                        if not ok_args:
                            result = {
                                "error": "invalid_tool_arguments",
                                "tool": tool_name,
                                "detail": arg_detail,
                                "got": arguments,
                            }
                            _dbg(f"tool_result name={tool_name} result={_shorten_text(result, 700)}")
                            messages.append(
                                ToolPromptMessage(
                                    tool_call_id=str(call_id or ""),
                                    name=tool_name,
                                    content=json.dumps(result, ensure_ascii=False),
                                )
                            )
                            messages.append(UserPromptMessage(content=_tool_call_retry_prompt(tool_name, arg_detail)))
                            continue

                        if tool_name in {"list_skill_files", "read_skill_file", "run_skill_command"}:
                            skill_name = str(arguments.get("skill_name") or "").strip()
                            if skill_name and not runtime.has_skill_metadata(skill_name):
                                result = {
                                    "error": "skill_md_required",
                                    "skill_name": skill_name,
                                    "detail": "必须先调用 get_skill_metadata(skill_name) 读取 SKILL.md（说明书）后，才能继续调用该工具。",
                                }
                                _dbg(f"tool_result name={tool_name} result={_shorten_text(result, 700)}")
                                messages.append(
                                    ToolPromptMessage(
                                        tool_call_id=str(call_id or ""),
                                        name=tool_name,
                                        content=json.dumps(result, ensure_ascii=False),
                                    )
                                )
                                messages.append(
                                    UserPromptMessage(
                                        content=(
                                            f"你刚才尝试调用 `{tool_name}` 但尚未读取技能《{skill_name}》的 SKILL.md。"
                                            f"请先调用 get_skill_metadata({skill_name!r})，再重试该工具调用。"
                                        )
                                    )
                                )
                                continue
                            if tool_name == "run_skill_command" and skill_name and not runtime.has_listed_skill_files(skill_name):
                                result = {
                                    "error": "skill_files_listing_required",
                                    "skill_name": skill_name,
                                    "detail": "执行技能命令前，必须先调用 list_skill_files(skill_name) 查看技能包目录结构。",
                                }
                                _dbg(f"tool_result name={tool_name} result={_shorten_text(result, 700)}")
                                messages.append(
                                    ToolPromptMessage(
                                        tool_call_id=str(call_id or ""),
                                        name=tool_name,
                                        content=json.dumps(result, ensure_ascii=False),
                                    )
                                )
                                messages.append(
                                    UserPromptMessage(
                                        content=(
                                            f"你刚才尝试调用 `{tool_name}` 但尚未查看技能《{skill_name}》的目录结构。"
                                            f"请先调用 list_skill_files({skill_name!r})，再重试该工具调用。"
                                        )
                                    )
                                )
                                continue

                        if tool_name == "get_skill_metadata":
                            yield self.create_text_message(
                                f"✅正在查看技能《{str(arguments.get('skill_name') or '')}》说明书…\n"
                            )
                        elif tool_name == "list_skill_files":
                            yield self.create_text_message(
                                f"✅正在查看技能《{str(arguments.get('skill_name') or '')}》文件结构…\n"
                            )
                        elif tool_name == "read_skill_file":
                            yield self.create_text_message(
                                f"✅正在读取技能《{str(arguments.get('skill_name') or '')}》文件：{str(arguments.get('relative_path') or '')}…\n"
                            )
                        elif tool_name == "run_skill_command":
                            yield self.create_text_message(
                                f"✅正在执行技能《{str(arguments.get('skill_name') or '')}》命令…\n"
                            )

                        if tool_name == "get_skill_metadata":
                            result = runtime.get_skill_metadata(str(arguments.get("skill_name") or ""))
                        elif tool_name == "list_skill_files":
                            result = runtime.list_skill_files(
                                str(arguments.get("skill_name") or ""),
                                int(arguments.get("max_depth") or 2),
                            )
                        elif tool_name == "read_skill_file":
                            result = runtime.read_skill_file(
                                str(arguments.get("skill_name") or ""),
                                str(arguments.get("relative_path") or ""),
                                int(arguments.get("max_chars") or 12000),
                            )
                        elif tool_name == "run_skill_command":
                            result = runtime.run_skill_command(
                                skill_name=str(arguments.get("skill_name") or ""),
                                command=arguments.get("command") if isinstance(arguments.get("command"), list) else [],
                                cwd_relative=(
                                    str(arguments.get("cwd_relative")) if arguments.get("cwd_relative") else None
                                ),
                                auto_install=bool(arguments.get("auto_install") or False),
                            )
                            if (
                                isinstance(result, dict)
                                and result.get("returncode") is not None
                                and int(result.get("returncode") or 0) != 0
                            ):
                                stderr = str(result.get("stderr") or "").strip()
                                if stderr:
                                    yield self.create_text_message(
                                        "❌命令执行失败（stderr）：\n" + _shorten_text(redact_user_visible_text(stderr), 1200) + "\n"
                                    )
                        else:
                            result = {"error": f"unknown tool: {tool_name}"}

                        _dbg(f"tool_result name={tool_name} result={_shorten_text(result, 700)}")
                        messages.append(
                            ToolPromptMessage(
                                tool_call_id=str(call_id or ""),
                                name=tool_name,
                                content=json.dumps(result, ensure_ascii=False),
                            )
                        )
                    continue

                json_text = _extract_first_json_object(res_text)
                action: dict[str, Any] | None = None
                if json_text:
                    try:
                        action = json.loads(json_text)
                    except Exception:
                        action = None
                _dbg(f"json_protocol detected={bool(action)} snippet={_shorten_text(json_text or '', 200)}")

                if not res_text and not action and not nontext:
                    empty_responses += 1
                    _dbg(f"empty_response_count={empty_responses}")
                    if empty_responses < 3:
                        messages.append(
                            UserPromptMessage(
                                content='你刚才没有输出任何内容。请继续完成任务：如果支持函数调用请调用工具；否则请输出 JSON：{"type":"final","content":"..."}'
                            )
                        )
                        continue
                    final_text = "模型连续返回空响应，未生成任何结果。"
                    break

                if not action or action.get("type") == "final":
                    if action and action.get("type") == "final":
                        final_text = str(action.get("content") or "")
                        _dbg(f"final_json content_len={len(final_text)}")
                    else:
                        final_text = res_text
                        _dbg(f"final_text content_len={len(final_text)}")
                        if streamed_any and final_text:
                            final_text_already_streamed = True
                    break

                if action.get("type") != "tool":
                    final_text = res_text
                    _dbg(f"final_non_tool type={action.get('type')!s} content_len={len(final_text)}")
                    break

                name = str(action.get("name") or "")
                arguments = action.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {}

                ok_args, arg_detail = _validate_tool_arguments(name, arguments)
                if not ok_args:
                    messages.append(UserPromptMessage(content=_tool_call_retry_prompt(name, arg_detail)))
                    result = {
                        "error": "invalid_tool_arguments",
                        "tool": name,
                        "detail": arg_detail,
                        "got": arguments,
                    }
                    _dbg(f"json_tool_result name={name} result={_shorten_text(result, 700)}")
                    messages.append(
                        AssistantPromptMessage(
                            content="TOOL_RESULT\n" + json.dumps({"name": name, "result": result}, ensure_ascii=False)
                        )
                    )
                    continue

                if name in {"list_skill_files", "read_skill_file", "run_skill_command"}:
                    skill_name = str(arguments.get("skill_name") or "").strip()
                    if skill_name and not runtime.has_skill_metadata(skill_name):
                        messages.append(
                            UserPromptMessage(
                                content=(
                                    f"你刚才尝试调用 `{name}` 但尚未读取技能《{skill_name}》的 SKILL.md。"
                                    f"请先调用 get_skill_metadata({skill_name!r})，再重试该工具调用。"
                                )
                            )
                        )
                        result = {
                            "error": "skill_md_required",
                            "skill_name": skill_name,
                            "detail": "必须先调用 get_skill_metadata(skill_name) 读取 SKILL.md（说明书）后，才能继续调用该工具。",
                        }
                        _dbg(f"json_tool_result name={name} result={_shorten_text(result, 700)}")
                        messages.append(
                            AssistantPromptMessage(
                                content="TOOL_RESULT\n" + json.dumps({"name": name, "result": result}, ensure_ascii=False)
                            )
                        )
                        continue
                    if name == "run_skill_command" and skill_name and not runtime.has_listed_skill_files(skill_name):
                        messages.append(
                            UserPromptMessage(
                                content=(
                                    f"你刚才尝试调用 `{name}` 但尚未查看技能《{skill_name}》的目录结构。"
                                    f"请先调用 list_skill_files({skill_name!r})，再重试该工具调用。"
                                )
                            )
                        )
                        result = {
                            "error": "skill_files_listing_required",
                            "skill_name": skill_name,
                            "detail": "执行技能命令前，必须先调用 list_skill_files(skill_name) 查看技能包目录结构。",
                        }
                        _dbg(f"json_tool_result name={name} result={_shorten_text(result, 700)}")
                        messages.append(
                            AssistantPromptMessage(
                                content="TOOL_RESULT\n" + json.dumps({"name": name, "result": result}, ensure_ascii=False)
                            )
                        )
                        continue

                _dbg(f"json_tool name={name} args={_shorten_text(arguments, 400)}")
                messages.append(AssistantPromptMessage(content=json.dumps(action, ensure_ascii=False)))

                if name == "get_skill_metadata":
                    yield self.create_text_message(f"✅正在查看技能《{str(arguments.get('skill_name') or '')}》说明书…\n")
                elif name == "list_skill_files":
                    yield self.create_text_message(f"✅正在查看技能《{str(arguments.get('skill_name') or '')}》文件结构…\n")
                elif name == "read_skill_file":
                    yield self.create_text_message(
                        f"✅正在读取技能《{str(arguments.get('skill_name') or '')}》文件：{str(arguments.get('relative_path') or '')}…\n"
                    )
                elif name == "run_skill_command":
                    yield self.create_text_message(
                        f"✅正在执行技能《{str(arguments.get('skill_name') or '')}》命令…\n"
                    )

                if name == "get_skill_metadata":
                    result = runtime.get_skill_metadata(str(arguments.get("skill_name") or ""))
                elif name == "list_skill_files":
                    result = runtime.list_skill_files(
                        str(arguments.get("skill_name") or ""),
                        int(arguments.get("max_depth") or 2),
                    )
                elif name == "read_skill_file":
                    result = runtime.read_skill_file(
                        str(arguments.get("skill_name") or ""),
                        str(arguments.get("relative_path") or ""),
                        int(arguments.get("max_chars") or 12000),
                    )
                elif name == "run_skill_command":
                    result = runtime.run_skill_command(
                        skill_name=str(arguments.get("skill_name") or ""),
                        command=arguments.get("command") if isinstance(arguments.get("command"), list) else [],
                        cwd_relative=(str(arguments.get("cwd_relative")) if arguments.get("cwd_relative") else None),
                        auto_install=bool(arguments.get("auto_install") or False),
                    )
                else:
                    result = {"error": f"unknown tool: {name}"}

                _dbg(f"json_tool_result name={name} result={_shorten_text(result, 700)}")
                messages.append(
                    AssistantPromptMessage(
                        content="TOOL_RESULT\n" + json.dumps({"name": name, "result": result}, ensure_ascii=False)
                    )
                )
            else:
                final_text = f"❌超过最大执行轮数 max_steps={max_steps}，仍未得到最终结果"
        finally:
            assistant_text_for_history = (final_text or "未生成任何文本输出。").strip()
            _append_history_turn(
                storage,
                history_key=history_key,
                user_text=user_input,
                assistant_text=assistant_text_for_history,
            )
            if final_text and final_text.strip() and not final_text_already_streamed:
                yield from stream_text_to_user(final_text)
            elif not final_text:
                yield from stream_text_to_user("未生成任何文本输出。")
