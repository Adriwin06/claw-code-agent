from __future__ import annotations

from typing import Sequence

from .cli.agent_cli_config import _build_agent, _build_runtime_config
from .cli.agent_runtime_ops import _AgentLiveRenderer, _run_agent_chat_loop
from .cli.command_dispatch import dispatch_main_command
from .cli.parser_build import build_parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return dispatch_main_command(args, parser=parser)


__all__ = [
    '_AgentLiveRenderer',
    '_build_runtime_config',
    '_build_agent',
    '_run_agent_chat_loop',
    'build_parser',
    'main',
]


if __name__ == '__main__':
    raise SystemExit(main())
