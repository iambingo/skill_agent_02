from __future__ import annotations

import os
import shlex
import subprocess
import sys
from typing import Any

from utils.skill_agent_constants import ALLOWED_COMMANDS
from utils.skill_agent_exec import (
    _ensure_python_module,
    _missing_executable_hint,
    _resolve_executable,
    _skill_contains_python_module,
)
from utils.skill_agent_paths import (
    _rewrite_uploads_paths_to_session_dir,
)
from utils.tools import _list_dir, _parse_frontmatter, _read_text, _safe_join


class _AgentRuntime:
    def __init__(
        self,
        *,
        skills_root: str | None,
        session_dir: str,
        max_steps: int,
        memory_turns: int,
    ) -> None:
        self.skills_root = skills_root
        self.session_dir = session_dir
        self.max_steps = max_steps
        self.memory_turns = memory_turns
        self._skill_metadata_cache: dict[str, dict[str, Any]] = {}
        self._skill_files_listed: set[str] = set()

    def has_skill_metadata(self, skill_name: str) -> bool:
        cached = self._skill_metadata_cache.get(skill_name)
        return bool(isinstance(cached, dict) and cached.get("skill") == skill_name)

    def load_skills_index(self) -> dict[str, Any]:
        if not self.skills_root:
            return {"root": None, "skills": []}
        skills: list[dict[str, Any]] = []
        for folder in sorted(os.listdir(self.skills_root)):
            path = os.path.join(self.skills_root, folder)
            if not os.path.isdir(path):
                continue
            skill_md = os.path.join(path, "SKILL.md")
            meta: dict[str, str] = {}
            if os.path.isfile(skill_md):
                meta = _parse_frontmatter(_read_text(skill_md, 4000))
            skills.append(
                {
                    "name": meta.get("name") or folder,
                    "folder": folder,
                    "description": meta.get("description") or "",
                }
            )
        return {"root": self.skills_root, "skills": skills}

    def get_skill_metadata(self, skill_name: str) -> dict[str, Any]:
        if not self.skills_root:
            return {"error": "skills_root not found"}
        path = _safe_join(self.skills_root, skill_name)
        skill_md = os.path.join(path, "SKILL.md")
        if not os.path.isfile(skill_md):
            return {"error": "SKILL.md not found", "skill": skill_name}
        content = _read_text(skill_md, 12000)
        meta = _parse_frontmatter(content)
        self._skill_metadata_cache[skill_name] = {"skill": skill_name, "metadata": meta}
        return {"skill": skill_name, "metadata": meta, "skill_md": content}

    def list_skill_files(self, skill_name: str, max_depth: int = 2) -> dict[str, Any]:
        if not self.skills_root:
            return {"error": "skills_root not found"}
        skill_path = _safe_join(self.skills_root, skill_name)
        self._skill_files_listed.add(skill_name)
        return {"skill": skill_name, "entries": _list_dir(skill_path, max_depth=max_depth)}

    def has_listed_skill_files(self, skill_name: str) -> bool:
        return str(skill_name or "").strip() in self._skill_files_listed

    def read_skill_file(self, skill_name: str, relative_path: str, max_chars: int = 12000) -> dict[str, Any]:
        if not self.skills_root:
            return {"error": "skills_root not found"}
        skill_path = _safe_join(self.skills_root, skill_name)
        file_path = _safe_join(skill_path, relative_path)
        if not os.path.isfile(file_path):
            return {"error": "file not found", "path": relative_path}
        return {"path": file_path, "content": _read_text(file_path, max_chars)}

    def run_skill_command(
        self,
        *,
        skill_name: str,
        command: list[str] | str,
        cwd_relative: str | None = None,
        auto_install: bool = False,
    ) -> dict[str, Any]:
        if not self.skills_root:
            return {"error": "skills_root not found"}
        if isinstance(command, str):
            command = shlex.split(command)
        if not command:
            return {"error": "command must be a non-empty list"}
        skill_path = _safe_join(self.skills_root, skill_name)
        exe = command[0]
        if exe == "python":
            if "-m" in command:
                module_index = command.index("-m") + 1
                if module_index < len(command):
                    module_name = command[module_index]
                    if not _skill_contains_python_module(skill_path, str(module_name)):
                        return {
                            "error": "no_executable_found",
                            "skill": skill_name,
                            "reason": "python -m module not found in skill folder",
                            "module": str(module_name),
                        }
                    module_check = _ensure_python_module(str(module_name), auto_install=auto_install, cwd=self.session_dir)
                    if not module_check.get("ok"):
                        return module_check
            command = [sys.executable] + command[1:]
        elif exe not in ALLOWED_COMMANDS:
            return {"error": f"command not allowed: {exe}"}
        resolved0 = _resolve_executable(str(command[0] or ""))
        if not resolved0:
            missing = str(command[0] or exe)
            return {"error": "executable_not_found", "exe": missing, "hint": _missing_executable_hint(missing)}
        command = [resolved0] + command[1:]
        command = _rewrite_uploads_paths_to_session_dir(command, session_dir=self.session_dir)
        # Make script path absolute relative to skill_path so cwd doesn't affect script resolution.
        # e.g. "python Scripts/foo.py" with cwd_relative="Scripts" would otherwise become Scripts/Scripts/foo.py
        if len(command) > 1 and not os.path.isabs(command[1]):
            candidate = os.path.normpath(os.path.join(skill_path, command[1]))
            if os.path.isfile(candidate):
                command = [command[0], candidate] + command[2:]
        # Normalize cwd_relative: strip leading slashes and ignore if it resolves to skill_path itself
        if cwd_relative:
            cwd_rel_norm = cwd_relative.strip("/").strip("\\")
            # If the LLM passed the skill name itself as cwd_relative, skip it (skill_path is already there)
            if cwd_rel_norm == skill_name or cwd_rel_norm == "":
                cwd_rel_norm = None
        else:
            cwd_rel_norm = None
        cwd = skill_path if not cwd_rel_norm else _safe_join(skill_path, cwd_rel_norm)
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
        except FileNotFoundError as e:
            return {"error": "executable_not_found", "exe": str(command[0] or exe), "exception": str(e)}
        except Exception as e:
            return {"error": "subprocess_failed", "exe": str(command[0] or exe), "exception": str(e)}

