from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from src.agent.models.types import AgentRuntimeConfig, ModelConfig


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str
    fix_hint: str | None = None


@dataclass(frozen=True)
class DoctorReport:
    workspace: str
    model: str
    base_url: str
    checks: tuple[DoctorCheck, ...] = field(default_factory=tuple)

    @property
    def has_failures(self) -> bool:
        return any(check.status == 'fail' for check in self.checks)

    def as_text(self) -> str:
        lines = [
            '# Doctor',
            '',
            f'workspace={self.workspace}',
            f'model={self.model}',
            f'base_url={self.base_url}',
            '',
        ]
        for check in self.checks:
            lines.append(f'[{check.status}] {check.name}: {check.detail}')
            if check.fix_hint:
                lines.append(f'  fix={check.fix_hint}')
        if not self.checks:
            lines.append('No checks were run.')
            return '\n'.join(lines)
        lines.extend(
            [
                '',
                '# Result',
                'ready=yes' if not self.has_failures else 'ready=no',
            ]
        )
        return '\n'.join(lines)


def run_doctor(
    *,
    model_config: ModelConfig,
    runtime_config: AgentRuntimeConfig,
    check_backend: bool = True,
    check_tui: bool = True,
) -> DoctorReport:
    checks = [
        _check_workspace(runtime_config.cwd),
        _check_session_storage(runtime_config),
        _check_git(runtime_config.cwd),
    ]
    if check_tui:
        checks.append(_check_textual())
    if check_backend:
        checks.extend(_check_backend(model_config))
    return DoctorReport(
        workspace=str(runtime_config.cwd.resolve()),
        model=model_config.model,
        base_url=model_config.base_url,
        checks=tuple(checks),
    )


def _check_workspace(cwd: Path) -> DoctorCheck:
    root = cwd.resolve()
    if not root.exists():
        return DoctorCheck(
            name='workspace',
            status='fail',
            detail=f'Workspace does not exist: {root}',
            fix_hint='Pass a valid --cwd path.',
        )
    if not root.is_dir():
        return DoctorCheck(
            name='workspace',
            status='fail',
            detail=f'Workspace is not a directory: {root}',
            fix_hint='Pass a directory path to --cwd.',
        )
    return DoctorCheck(
        name='workspace',
        status='ok',
        detail=f'Using workspace {root}',
    )


def _check_session_storage(runtime_config: AgentRuntimeConfig) -> DoctorCheck:
    targets = (
        runtime_config.session_directory.resolve(),
        runtime_config.scratchpad_root.resolve(),
    )
    try:
        for directory in targets:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix='doctor_',
                suffix='.tmp',
                dir=directory,
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                handle.write(b'ok')
            tmp_path.unlink(missing_ok=True)
    except OSError as exc:
        return DoctorCheck(
            name='session-storage',
            status='fail',
            detail=f'Unable to write to session storage: {exc}',
            fix_hint='Check workspace permissions or override the session directories.',
        )
    return DoctorCheck(
        name='session-storage',
        status='ok',
        detail='Session and scratchpad directories are writable.',
    )


def _check_git(cwd: Path) -> DoctorCheck:
    git_path = shutil.which('git')
    if git_path is None:
        return DoctorCheck(
            name='git',
            status='warn',
            detail='git is not available on PATH.',
            fix_hint='Install git to enable worktree-aware and repository-aware flows.',
        )
    if not (cwd.resolve() / '.git').exists():
        return DoctorCheck(
            name='git',
            status='warn',
            detail=f'git is installed at {git_path}, but this workspace is not a git checkout.',
            fix_hint='Run the agent from a repository root for the best coding workflow.',
        )
    return DoctorCheck(
        name='git',
        status='ok',
        detail=f'git is available at {git_path}.',
    )


def _check_textual() -> DoctorCheck:
    if not _module_available('textual.app'):
        return DoctorCheck(
            name='tui',
            status='warn',
            detail='Textual is not installed; agent-tui will not launch.',
            fix_hint='Install the optional TUI extra with `pip install -e .[tui]`.',
        )
    if not _module_available('PIL.Image'):
        return DoctorCheck(
            name='tui',
            status='warn',
            detail='Textual is installed, but Pillow is missing; display_image previews will fall back to paths.',
            fix_hint='Install the optional TUI extra with `pip install -e .[tui]`, or rebuild the Docker image.',
        )
    return DoctorCheck(
        name='tui',
        status='ok',
        detail='Textual and Pillow are installed.',
    )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _check_backend(model_config: ModelConfig) -> tuple[DoctorCheck, ...]:
    models_url = model_config.base_url.rstrip('/') + '/models'
    req = urllib_request.Request(
        models_url,
        headers={
            'Authorization': f'Bearer {model_config.api_key}',
            'Accept': 'application/json',
        },
        method='GET',
    )
    try:
        with urllib_request.urlopen(req, timeout=model_config.timeout_seconds) as response:
            payload = json.loads(response.read().decode('utf-8', errors='replace'))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace').strip()
        if not detail:
            detail = f'HTTP {exc.code}'
        return (
            DoctorCheck(
                name='backend',
                status='fail',
                detail=f'Backend check failed against {models_url}: {detail}',
                fix_hint='Start the OpenAI-compatible model server or update --base-url / --api-key.',
            ),
        )
    except urllib_error.URLError as exc:
        reason = exc.reason if isinstance(exc.reason, str) else str(exc.reason)
        return (
            DoctorCheck(
                name='backend',
                status='fail',
                detail=f'Unable to reach {models_url}: {reason}',
                fix_hint='Start the model server or point --base-url at a reachable endpoint.',
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (
            DoctorCheck(
                name='backend',
                status='fail',
                detail=f'Backend response from {models_url} was invalid: {exc}',
                fix_hint='Verify that the endpoint is OpenAI-compatible and returning JSON.',
            ),
        )

    checks = [
        DoctorCheck(
            name='backend',
            status='ok',
            detail=f'Reached model backend at {models_url}.',
        )
    ]
    model_ids = _extract_model_ids(payload)
    if model_ids:
        if model_config.model in model_ids:
            checks.append(
                DoctorCheck(
                    name='model',
                    status='ok',
                    detail=f'Configured model `{model_config.model}` is advertised by the backend.',
                )
            )
        else:
            sample = ', '.join(model_ids[:5])
            checks.append(
                DoctorCheck(
                    name='model',
                    status='warn',
                    detail=(
                        f'Backend responded, but `{model_config.model}` was not listed.'
                        + (f' Available models: {sample}' if sample else '')
                    ),
                    fix_hint='Use --model with one of the advertised ids, or confirm that the backend aliases the configured model name.',
                )
            )
    else:
        checks.append(
            DoctorCheck(
                name='model',
                status='warn',
                detail='Backend responded, but no model ids were returned by /models.',
                fix_hint='Verify the backend supports the OpenAI /models endpoint.',
            )
        )
    return tuple(checks)


def _extract_model_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get('data')
    if not isinstance(data, list):
        return []
    model_ids: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get('id')
        if isinstance(model_id, str) and model_id.strip():
            model_ids.append(model_id.strip())
    return model_ids
