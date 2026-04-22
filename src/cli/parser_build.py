from __future__ import annotations

import argparse

from .agent_cli_config import (
    _add_agent_common_args,
    _add_agent_resume_args,
    _default_model_from_env,
    _env_first,
)


def _add_internal_surface_parsers(subparsers, *, hidden: bool) -> None:
    def _help(text: str) -> str:
        return argparse.SUPPRESS if hidden else text

    subparsers.add_parser('summary', help=_help('render a Markdown summary of the porting workspace'))
    subparsers.add_parser('manifest', help=_help('print the current workspace manifest'))
    subparsers.add_parser(
        'parity-audit',
        help=_help('compare the Python workspace against the local ignored TypeScript archive when available'),
    )
    subparsers.add_parser('setup-report', help=_help('render the startup and prefetch setup report'))
    subparsers.add_parser('command-graph', help=_help('show command graph segmentation'))
    subparsers.add_parser('tool-pool', help=_help('show the assembled tool pool with default settings'))
    subparsers.add_parser(
        'bootstrap-graph',
        help=_help('show the mirrored bootstrap and runtime graph stages'),
    )

    list_parser = subparsers.add_parser(
        'subsystems',
        help=_help('list the current Python modules in the workspace'),
    )
    list_parser.add_argument('--limit', type=int, default=32)

    commands_parser = subparsers.add_parser(
        'commands',
        help=_help('list mirrored command entries from the archived snapshot'),
    )
    commands_parser.add_argument('--limit', type=int, default=20)
    commands_parser.add_argument('--query')
    commands_parser.add_argument('--no-plugin-commands', action='store_true')
    commands_parser.add_argument('--no-skill-commands', action='store_true')

    tools_parser = subparsers.add_parser(
        'tools',
        help=_help('list mirrored tool entries from the archived snapshot'),
    )
    tools_parser.add_argument('--limit', type=int, default=20)
    tools_parser.add_argument('--query')
    tools_parser.add_argument('--simple-mode', action='store_true')
    tools_parser.add_argument('--no-mcp', action='store_true')
    tools_parser.add_argument('--deny-tool', action='append', default=[])
    tools_parser.add_argument('--deny-prefix', action='append', default=[])

    route_parser = subparsers.add_parser(
        'route',
        help=_help('route a prompt across mirrored command and tool inventories'),
    )
    route_parser.add_argument('prompt')
    route_parser.add_argument('--limit', type=int, default=5)

    bootstrap_parser = subparsers.add_parser(
        'bootstrap',
        help=_help('build a runtime-style session report from the mirrored inventories'),
    )
    bootstrap_parser.add_argument('prompt')
    bootstrap_parser.add_argument('--limit', type=int, default=5)

    loop_parser = subparsers.add_parser(
        'turn-loop',
        help=_help('run a small stateful turn loop for the mirrored runtime'),
    )
    loop_parser.add_argument('prompt')
    loop_parser.add_argument('--limit', type=int, default=5)
    loop_parser.add_argument('--max-turns', type=int, default=3)
    loop_parser.add_argument('--structured-output', action='store_true')

    flush_parser = subparsers.add_parser(
        'flush-transcript',
        help=_help('persist and flush a temporary session transcript'),
    )
    flush_parser.add_argument('prompt')

    load_session_parser = subparsers.add_parser(
        'load-session',
        help=_help('load a previously persisted session'),
    )
    load_session_parser.add_argument('session_id')

    for name, help_text in (
        ('remote-mode', 'simulate remote-control runtime branching'),
        ('ssh-mode', 'simulate SSH runtime branching'),
        ('teleport-mode', 'simulate teleport runtime branching'),
        ('direct-connect-mode', 'simulate direct-connect runtime branching'),
        ('deep-link-mode', 'simulate deep-link runtime branching'),
    ):
        parser = subparsers.add_parser(name, help=_help(help_text))
        parser.add_argument('target')
        parser.add_argument('--cwd', default='.')

    show_command = subparsers.add_parser(
        'show-command',
        help=_help('show one mirrored command entry by exact name'),
    )
    show_command.add_argument('name')

    show_tool = subparsers.add_parser(
        'show-tool',
        help=_help('show one mirrored tool entry by exact name'),
    )
    show_tool.add_argument('name')

    exec_command_parser = subparsers.add_parser(
        'exec-command',
        help=_help('execute a mirrored command shim by exact name'),
    )
    exec_command_parser.add_argument('name')
    exec_command_parser.add_argument('prompt')

    exec_tool_parser = subparsers.add_parser(
        'exec-tool',
        help=_help('execute a mirrored tool shim by exact name'),
    )
    exec_tool_parser.add_argument('name')
    exec_tool_parser.add_argument('payload')


def _hide_suppressed_subparser_actions(subparsers) -> None:
    visible_actions = [
        action
        for action in getattr(subparsers, '_choices_actions', [])
        if action.help != argparse.SUPPRESS
    ]
    subparsers._choices_actions = visible_actions
    visible_names = [action.dest for action in visible_actions if isinstance(action.dest, str)]
    if visible_names:
        subparsers.metavar = '{' + ','.join(visible_names) + '}'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Local coding agent with runtime and developer utilities')
    subparsers = parser.add_subparsers(dest='command', required=True)
    _add_internal_surface_parsers(subparsers, hidden=True)

    doctor_parser = subparsers.add_parser(
        'doctor',
        help='check workspace, backend, and optional UI readiness',
    )
    doctor_parser.add_argument('--model', default=_default_model_from_env())
    doctor_parser.add_argument(
        '--base-url',
        default=_env_first('LLM_API_BASE', default='http://127.0.0.1:8000/v1'),
    )
    doctor_parser.add_argument(
        '--api-key',
        default=None,
    )
    doctor_parser.add_argument('--timeout-seconds', type=float, default=15.0)
    doctor_parser.add_argument('--cwd', default='.')
    doctor_parser.add_argument('--skip-backend', action='store_true')
    doctor_parser.add_argument('--skip-tui', action='store_true')

    dev_parser = subparsers.add_parser('dev', help='developer and porting utilities')
    dev_subparsers = dev_parser.add_subparsers(dest='dev_command')
    dev_subparsers.required = True
    _add_internal_surface_parsers(dev_subparsers, hidden=False)
    remote_status_parser = subparsers.add_parser('remote-status', help='show local remote runtime status')
    remote_status_parser.add_argument('--cwd', default='.')
    remote_profiles_parser = subparsers.add_parser('remote-profiles', help='list configured local remote profiles')
    remote_profiles_parser.add_argument('--cwd', default='.')
    remote_profiles_parser.add_argument('--query')
    remote_disconnect_parser = subparsers.add_parser('remote-disconnect', help='disconnect the active local remote target')
    remote_disconnect_parser.add_argument('--cwd', default='.')
    worktree_status_parser = subparsers.add_parser('worktree-status', help='show local managed git worktree status')
    worktree_status_parser.add_argument('--cwd', default='.')
    worktree_enter_parser = subparsers.add_parser('worktree-enter', help='create and enter a managed git worktree')
    worktree_enter_parser.add_argument('name', nargs='?')
    worktree_enter_parser.add_argument('--cwd', default='.')
    worktree_exit_parser = subparsers.add_parser('worktree-exit', help='exit the active managed git worktree')
    worktree_exit_parser.add_argument('--action', default='keep')
    worktree_exit_parser.add_argument('--discard-changes', action='store_true')
    worktree_exit_parser.add_argument('--cwd', default='.')
    account_status_parser = subparsers.add_parser('account-status', help='show local account runtime status')
    account_status_parser.add_argument('--cwd', default='.')
    account_profiles_parser = subparsers.add_parser('account-profiles', help='list configured local account profiles')
    account_profiles_parser.add_argument('--cwd', default='.')
    account_profiles_parser.add_argument('--query')
    account_login_parser = subparsers.add_parser('account-login', help='activate a local account profile or ephemeral identity')
    account_login_parser.add_argument('target')
    account_login_parser.add_argument('--provider')
    account_login_parser.add_argument('--auth-mode')
    account_login_parser.add_argument('--cwd', default='.')
    account_logout_parser = subparsers.add_parser('account-logout', help='clear the active local account session')
    account_logout_parser.add_argument('--cwd', default='.')
    ask_status_parser = subparsers.add_parser('ask-status', help='show local ask-user runtime status')
    ask_status_parser.add_argument('--cwd', default='.')
    ask_history_parser = subparsers.add_parser('ask-history', help='show local ask-user interaction history')
    ask_history_parser.add_argument('--cwd', default='.')
    search_status_parser = subparsers.add_parser('search-status', help='show local search runtime status')
    search_status_parser.add_argument('--cwd', default='.')
    search_status_parser.add_argument('--provider')
    search_providers_parser = subparsers.add_parser('search-providers', help='list configured local search providers')
    search_providers_parser.add_argument('--cwd', default='.')
    search_providers_parser.add_argument('--query')
    search_activate_parser = subparsers.add_parser('search-activate', help='set the active local search provider')
    search_activate_parser.add_argument('provider')
    search_activate_parser.add_argument('--cwd', default='.')
    search_parser = subparsers.add_parser('search', help='run a real web search against the configured local search runtime')
    search_parser.add_argument('query')
    search_parser.add_argument('--cwd', default='.')
    search_parser.add_argument('--provider')
    search_parser.add_argument('--max-results', type=int, default=5)
    search_parser.add_argument('--domain', action='append', default=[])
    mcp_status_parser = subparsers.add_parser('mcp-status', help='show local MCP runtime status')
    mcp_status_parser.add_argument('--cwd', default='.')
    mcp_resources_parser = subparsers.add_parser('mcp-resources', help='list MCP resources discovered through local manifests and transport-backed servers')
    mcp_resources_parser.add_argument('--cwd', default='.')
    mcp_resources_parser.add_argument('--query')
    mcp_resource_parser = subparsers.add_parser('mcp-resource', help='read an MCP resource by URI')
    mcp_resource_parser.add_argument('uri')
    mcp_resource_parser.add_argument('--cwd', default='.')
    mcp_tools_parser = subparsers.add_parser('mcp-tools', help='list MCP tools exposed by configured MCP servers')
    mcp_tools_parser.add_argument('--cwd', default='.')
    mcp_tools_parser.add_argument('--query')
    mcp_tools_parser.add_argument('--server')
    mcp_call_tool_parser = subparsers.add_parser('mcp-call-tool', help='call an MCP tool exposed by a configured MCP server')
    mcp_call_tool_parser.add_argument('tool_name')
    mcp_call_tool_parser.add_argument('--arguments-json', default='{}')
    mcp_call_tool_parser.add_argument('--server')
    mcp_call_tool_parser.add_argument('--cwd', default='.')
    config_status_parser = subparsers.add_parser('config-status', help='show local workspace config runtime summary')
    config_status_parser.add_argument('--cwd', default='.')
    config_effective_parser = subparsers.add_parser('config-effective', help='render the merged effective local workspace config')
    config_effective_parser.add_argument('--cwd', default='.')
    config_source_parser = subparsers.add_parser('config-source', help='render a specific local config source')
    config_source_parser.add_argument('source')
    config_source_parser.add_argument('--cwd', default='.')
    config_get_parser = subparsers.add_parser('config-get', help='read a local config value by dotted key path')
    config_get_parser.add_argument('key_path')
    config_get_parser.add_argument('--source')
    config_get_parser.add_argument('--cwd', default='.')
    config_set_parser = subparsers.add_parser('config-set', help='write a local config value by dotted key path')
    config_set_parser.add_argument('key_path')
    config_set_parser.add_argument('value_json')
    config_set_parser.add_argument('--source', default='local')
    config_set_parser.add_argument('--cwd', default='.')
    lsp_status_parser = subparsers.add_parser('lsp-status', help='show local LSP runtime summary')
    lsp_status_parser.add_argument('--cwd', default='.')
    lsp_symbols_parser = subparsers.add_parser('lsp-symbols', help='show local LSP document symbols for one file')
    lsp_symbols_parser.add_argument('file_path')
    lsp_symbols_parser.add_argument('--cwd', default='.')
    lsp_workspace_parser = subparsers.add_parser('lsp-workspace-symbols', help='search workspace symbols through the local LSP runtime')
    lsp_workspace_parser.add_argument('query')
    lsp_workspace_parser.add_argument('--max-results', type=int, default=50)
    lsp_workspace_parser.add_argument('--cwd', default='.')
    lsp_definition_parser = subparsers.add_parser('lsp-definition', help='run a local LSP definition query')
    lsp_definition_parser.add_argument('file_path')
    lsp_definition_parser.add_argument('line', type=int)
    lsp_definition_parser.add_argument('character', type=int)
    lsp_definition_parser.add_argument('--max-results', type=int, default=20)
    lsp_definition_parser.add_argument('--cwd', default='.')
    lsp_references_parser = subparsers.add_parser('lsp-references', help='run a local LSP references query')
    lsp_references_parser.add_argument('file_path')
    lsp_references_parser.add_argument('line', type=int)
    lsp_references_parser.add_argument('character', type=int)
    lsp_references_parser.add_argument('--max-results', type=int, default=50)
    lsp_references_parser.add_argument('--cwd', default='.')
    lsp_hover_parser = subparsers.add_parser('lsp-hover', help='run a local LSP hover query')
    lsp_hover_parser.add_argument('file_path')
    lsp_hover_parser.add_argument('line', type=int)
    lsp_hover_parser.add_argument('character', type=int)
    lsp_hover_parser.add_argument('--cwd', default='.')
    lsp_diagnostics_parser = subparsers.add_parser('lsp-diagnostics', help='show local LSP diagnostics')
    lsp_diagnostics_parser.add_argument('--file-path')
    lsp_diagnostics_parser.add_argument('--cwd', default='.')
    lsp_hierarchy_parser = subparsers.add_parser('lsp-call-hierarchy', help='show local LSP call hierarchy at a position')
    lsp_hierarchy_parser.add_argument('file_path')
    lsp_hierarchy_parser.add_argument('line', type=int)
    lsp_hierarchy_parser.add_argument('character', type=int)
    lsp_hierarchy_parser.add_argument('--cwd', default='.')
    lsp_incoming_parser = subparsers.add_parser('lsp-incoming-calls', help='show local LSP incoming calls at a position')
    lsp_incoming_parser.add_argument('file_path')
    lsp_incoming_parser.add_argument('line', type=int)
    lsp_incoming_parser.add_argument('character', type=int)
    lsp_incoming_parser.add_argument('--max-results', type=int, default=50)
    lsp_incoming_parser.add_argument('--cwd', default='.')
    lsp_outgoing_parser = subparsers.add_parser('lsp-outgoing-calls', help='show local LSP outgoing calls at a position')
    lsp_outgoing_parser.add_argument('file_path')
    lsp_outgoing_parser.add_argument('line', type=int)
    lsp_outgoing_parser.add_argument('character', type=int)
    lsp_outgoing_parser.add_argument('--max-results', type=int, default=50)
    lsp_outgoing_parser.add_argument('--cwd', default='.')
    workflow_list_parser = subparsers.add_parser('workflow-list', help='list local workflow definitions')
    workflow_list_parser.add_argument('--cwd', default='.')
    workflow_list_parser.add_argument('--query')
    workflow_get_parser = subparsers.add_parser('workflow-get', help='show one local workflow definition')
    workflow_get_parser.add_argument('workflow_name')
    workflow_get_parser.add_argument('--cwd', default='.')
    workflow_run_parser = subparsers.add_parser('workflow-run', help='record and render a local workflow run')
    workflow_run_parser.add_argument('workflow_name')
    workflow_run_parser.add_argument('--arguments-json', default='{}')
    workflow_run_parser.add_argument('--cwd', default='.')
    trigger_list_parser = subparsers.add_parser('trigger-list', help='list local remote triggers')
    trigger_list_parser.add_argument('--cwd', default='.')
    trigger_list_parser.add_argument('--query')
    trigger_get_parser = subparsers.add_parser('trigger-get', help='show one local remote trigger')
    trigger_get_parser.add_argument('trigger_id')
    trigger_get_parser.add_argument('--cwd', default='.')
    trigger_create_parser = subparsers.add_parser('trigger-create', help='create a local remote trigger')
    trigger_create_parser.add_argument('--body-json', required=True)
    trigger_create_parser.add_argument('--cwd', default='.')
    trigger_update_parser = subparsers.add_parser('trigger-update', help='update a local remote trigger')
    trigger_update_parser.add_argument('trigger_id')
    trigger_update_parser.add_argument('--body-json', required=True)
    trigger_update_parser.add_argument('--cwd', default='.')
    trigger_run_parser = subparsers.add_parser('trigger-run', help='run a local remote trigger')
    trigger_run_parser.add_argument('trigger_id')
    trigger_run_parser.add_argument('--body-json', default='{}')
    trigger_run_parser.add_argument('--cwd', default='.')
    teams_status_parser = subparsers.add_parser('team-status', help='show local collaboration team runtime summary')
    teams_status_parser.add_argument('--cwd', default='.')
    teams_list_parser = subparsers.add_parser('team-list', help='list local collaboration teams')
    teams_list_parser.add_argument('--cwd', default='.')
    teams_list_parser.add_argument('--query')
    team_get_parser = subparsers.add_parser('team-get', help='show one local collaboration team')
    team_get_parser.add_argument('team_name')
    team_get_parser.add_argument('--cwd', default='.')
    team_create_parser = subparsers.add_parser('team-create', help='create a local collaboration team')
    team_create_parser.add_argument('team_name')
    team_create_parser.add_argument('--description')
    team_create_parser.add_argument('--member', action='append', default=[])
    team_create_parser.add_argument('--cwd', default='.')
    team_delete_parser = subparsers.add_parser('team-delete', help='delete a local collaboration team')
    team_delete_parser.add_argument('team_name')
    team_delete_parser.add_argument('--cwd', default='.')
    team_messages_parser = subparsers.add_parser('team-messages', help='show local team messages')
    team_messages_parser.add_argument('--team-name')
    team_messages_parser.add_argument('--cwd', default='.')

    agent_parser = subparsers.add_parser('agent', help='run the real Python local-model agent')
    agent_parser.add_argument('prompt')
    agent_parser.add_argument('--max-turns', type=int)
    agent_parser.add_argument('--show-transcript', action='store_true')
    _add_agent_common_args(agent_parser, include_backend=True)

    background_parser = subparsers.add_parser('agent-bg', help='run the Python local-model agent as a local background session')
    background_parser.add_argument('prompt')
    background_parser.add_argument('--max-turns', type=int)
    background_parser.add_argument('--show-transcript', action='store_true')
    _add_agent_common_args(background_parser, include_backend=True)

    background_worker_parser = subparsers.add_parser('agent-bg-worker', help=argparse.SUPPRESS)
    background_worker_parser.add_argument('background_id')
    background_worker_parser.add_argument('prompt')
    background_worker_parser.add_argument('--background-root', required=True)
    background_worker_parser.add_argument('--max-turns', type=int)
    background_worker_parser.add_argument('--show-transcript', action='store_true')
    _add_agent_common_args(background_worker_parser, include_backend=True)

    ps_parser = subparsers.add_parser('agent-ps', help='list local background agent sessions')
    ps_parser.add_argument('--tail', type=int, default=None)

    logs_parser = subparsers.add_parser('agent-logs', help='show logs for a local background agent session')
    logs_parser.add_argument('background_id')
    logs_parser.add_argument('--tail', type=int, default=None)

    attach_parser = subparsers.add_parser('agent-attach', help='show the current output snapshot for a local background agent session')
    attach_parser.add_argument('background_id')
    attach_parser.add_argument('--tail', type=int, default=None)

    kill_parser = subparsers.add_parser('agent-kill', help='stop a local background agent session')
    kill_parser.add_argument('background_id')

    daemon_parser = subparsers.add_parser('daemon', help='manage local daemon-style background agent sessions')
    daemon_subparsers = daemon_parser.add_subparsers(dest='daemon_command')
    daemon_subparsers.required = True

    daemon_start_parser = daemon_subparsers.add_parser('start', help='launch a local daemon-style background agent session')
    daemon_start_parser.add_argument('prompt')
    daemon_start_parser.add_argument('--max-turns', type=int)
    daemon_start_parser.add_argument('--show-transcript', action='store_true')
    _add_agent_common_args(daemon_start_parser, include_backend=True)

    daemon_worker_parser = daemon_subparsers.add_parser('worker', help=argparse.SUPPRESS)
    daemon_worker_parser.add_argument('background_id')
    daemon_worker_parser.add_argument('prompt')
    daemon_worker_parser.add_argument('--background-root', required=True)
    daemon_worker_parser.add_argument('--max-turns', type=int)
    daemon_worker_parser.add_argument('--show-transcript', action='store_true')
    _add_agent_common_args(daemon_worker_parser, include_backend=True)

    daemon_ps_parser = daemon_subparsers.add_parser('ps', help='list local daemon-style background sessions')
    daemon_ps_parser.add_argument('--tail', type=int, default=None)

    daemon_logs_parser = daemon_subparsers.add_parser('logs', help='show logs for a local daemon-style background session')
    daemon_logs_parser.add_argument('background_id')
    daemon_logs_parser.add_argument('--tail', type=int, default=None)

    daemon_attach_parser = daemon_subparsers.add_parser('attach', help='show the current output snapshot for a local daemon-style background session')
    daemon_attach_parser.add_argument('background_id')
    daemon_attach_parser.add_argument('--tail', type=int, default=None)

    daemon_kill_parser = daemon_subparsers.add_parser('kill', help='stop a local daemon-style background session')
    daemon_kill_parser.add_argument('background_id')

    chat_parser = subparsers.add_parser('agent-chat', help='run an interactive Python local-model chat loop')
    chat_parser.add_argument('prompt', nargs='?')
    chat_parser.add_argument('--resume-session-id')
    chat_parser.add_argument('--max-turns', type=int)
    chat_parser.add_argument('--show-transcript', action='store_true')
    _add_agent_common_args(chat_parser, include_backend=True)

    tui_parser = subparsers.add_parser('agent-tui', help='run the Textual terminal UI for the local-model agent')
    tui_parser.add_argument('prompt', nargs='?')
    tui_parser.add_argument('--resume-session-id')
    tui_parser.add_argument('--max-turns', type=int)
    _add_agent_common_args(tui_parser, include_backend=True)

    resume_parser = subparsers.add_parser('agent-resume', help='resume a saved Python local-model agent session')
    _add_agent_resume_args(resume_parser)

    prompt_parser = subparsers.add_parser('agent-prompt', help='render the Python agent system prompt')
    _add_agent_common_args(prompt_parser, include_backend=False)

    context_parser = subparsers.add_parser('agent-context', help='render Python /context-style usage accounting')
    _add_agent_common_args(context_parser, include_backend=False)

    context_raw_parser = subparsers.add_parser('agent-context-raw', help='render the raw Python agent context snapshot')
    _add_agent_common_args(context_raw_parser, include_backend=False)

    token_budget_parser = subparsers.add_parser('token-budget', help='render the current token budget and prompt-length limits')
    _add_agent_common_args(token_budget_parser, include_backend=False)

    agents_parser = subparsers.add_parser('agents', help='list active local agent configurations or show one agent definition')
    agents_parser.add_argument('agent_type', nargs='?')
    agents_parser.add_argument('--all', action='store_true')
    _add_agent_common_args(agents_parser, include_backend=False)
    _hide_suppressed_subparser_actions(subparsers)
    return parser
