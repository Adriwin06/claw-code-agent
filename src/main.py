from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from .cli.agent_cli_config import _build_agent, _build_runtime_config
from .cli.agent_runtime_ops import _AgentLiveRenderer, _run_agent_chat_loop
from .cli.command_dispatch import dispatch_main_command
from .cli.parser_build import build_parser
from .features.system.env_runtime import load_workspace_env


def _detect_workspace_cwd(argv: Sequence[str]) -> Path:
    for index in range(len(argv) - 1, -1, -1):
        token = argv[index]
        if token == '--cwd' and index + 1 < len(argv):
            raw_value = argv[index + 1].strip()
            if raw_value:
                return Path(raw_value).resolve()
        if token.startswith('--cwd='):
            raw_value = token.partition('=')[2].strip()
            if raw_value:
                return Path(raw_value).resolve()
    return Path.cwd().resolve()


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    load_workspace_env(_detect_workspace_cwd(argv_list))
    parser = build_parser()
    args = parser.parse_args(argv_list)
    load_workspace_env(Path(args.cwd).resolve())
    return dispatch_main_command(args, parser=parser)


__all__ = [
    '_AgentLiveRenderer',
    '_build_runtime_config',
    '_build_agent',
    '_run_agent_chat_loop',
    '_detect_workspace_cwd',
    'build_parser',
    'main',
]


if __name__ == '__main__':
    raise SystemExit(main())
