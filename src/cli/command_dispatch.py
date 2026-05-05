from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from src.agent.models.types import AgentRuntimeConfig, ModelConfig
from src.core.catalog.catalog_runtime import (
    assemble_tool_pool,
    build_command_graph,
    build_port_manifest,
    execute_command,
    execute_tool,
    get_command,
    get_commands,
    get_tool,
    get_tools,
    render_command_index,
    render_tool_index,
)
from src.core.governance.permissions import ToolPermissionContext
from src.core.orchestration.bootstrap_runtime import build_bootstrap_graph, run_setup
from src.core.orchestration.query_engine import QueryEnginePort
from src.core.orchestration.runtime import PortRuntime
from src.core.catalog.parity_audit import run_parity_audit
from src.features.system.account_runtime import AccountRuntime
from src.features.collaboration.ask_user_runtime import AskUserRuntime
from src.features.system.background_runtime import BackgroundSessionRuntime
from src.features.system.config_runtime import ConfigRuntime
from src.features.system.doctor_runtime import run_doctor
from src.features.integration.lsp_runtime import LSPRuntime
from src.features.integration.mcp_runtime import MCPRuntime
from src.features.integration.remote_runtime import (
    RemoteRuntime,
    run_deep_link_mode,
    run_direct_connect_mode,
    run_remote_mode,
    run_ssh_mode,
    run_teleport_mode,
)
from src.features.integration.remote_trigger_runtime import RemoteTriggerRuntime
from src.features.integration.search_runtime import SearchRuntime
from src.features.collaboration.team_runtime import TeamRuntime
from src.features.orchestration.workflow_runtime import WorkflowRuntime
from src.features.orchestration.worktree_runtime import WorktreeRuntime
from src.session.session_store import load_session
from src.textual_ui import run_agent_tui
from .agent_cli_config import _build_agent, _env_nonempty, _resolve_api_key
from .agent_runtime_ops import (
    _build_resumed_agent,
    _launch_background_agent,
    _run_agent_chat_loop,
    _run_agent_turn,
    _run_background_worker,
)


CommandHandler = Callable[[argparse.Namespace], int]


def _cwd(args: argparse.Namespace) -> Path:
    return Path(args.cwd).resolve()


def _resolve_command_name(args: argparse.Namespace) -> str:
    if args.command == 'dev':
        return str(args.dev_command)
    return str(args.command)


def dispatch_main_command(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
) -> int:
    command_name = _resolve_command_name(args)
    handler = _build_command_handlers().get(command_name)
    if handler is None:
        parser.error(f'unknown command: {command_name}')
        return 2
    return handler(args)


def _build_command_handlers() -> dict[str, CommandHandler]:
    return {
        'doctor': _handle_doctor,
        'summary': _handle_summary,
        'manifest': _handle_manifest,
        'parity-audit': _handle_parity_audit,
        'setup-report': _handle_setup_report,
        'command-graph': _handle_command_graph,
        'tool-pool': _handle_tool_pool,
        'bootstrap-graph': _handle_bootstrap_graph,
        'subsystems': _handle_subsystems,
        'commands': _handle_commands,
        'tools': _handle_tools,
        'route': _handle_route,
        'bootstrap': _handle_bootstrap,
        'turn-loop': _handle_turn_loop,
        'flush-transcript': _handle_flush_transcript,
        'load-session': _handle_load_session,
        'remote-mode': _handle_remote_mode,
        'ssh-mode': _handle_ssh_mode,
        'teleport-mode': _handle_teleport_mode,
        'direct-connect-mode': _handle_direct_connect_mode,
        'deep-link-mode': _handle_deep_link_mode,
        'remote-status': _handle_remote_status,
        'remote-profiles': _handle_remote_profiles,
        'remote-disconnect': _handle_remote_disconnect,
        'worktree-status': _handle_worktree_status,
        'worktree-enter': _handle_worktree_enter,
        'worktree-exit': _handle_worktree_exit,
        'account-status': _handle_account_status,
        'account-profiles': _handle_account_profiles,
        'account-login': _handle_account_login,
        'account-logout': _handle_account_logout,
        'ask-status': _handle_ask_status,
        'ask-history': _handle_ask_history,
        'search-status': _handle_search_status,
        'search-providers': _handle_search_providers,
        'search-activate': _handle_search_activate,
        'search': _handle_search,
        'mcp-status': _handle_mcp_status,
        'mcp-resources': _handle_mcp_resources,
        'mcp-resource': _handle_mcp_resource,
        'mcp-tools': _handle_mcp_tools,
        'mcp-call-tool': _handle_mcp_call_tool,
        'config-status': _handle_config_status,
        'config-effective': _handle_config_effective,
        'config-source': _handle_config_source,
        'config-get': _handle_config_get,
        'config-set': _handle_config_set,
        'lsp-status': _handle_lsp_status,
        'lsp-symbols': _handle_lsp_symbols,
        'lsp-workspace-symbols': _handle_lsp_workspace_symbols,
        'lsp-definition': _handle_lsp_definition,
        'lsp-references': _handle_lsp_references,
        'lsp-hover': _handle_lsp_hover,
        'lsp-diagnostics': _handle_lsp_diagnostics,
        'lsp-call-hierarchy': _handle_lsp_call_hierarchy,
        'lsp-incoming-calls': _handle_lsp_incoming_calls,
        'lsp-outgoing-calls': _handle_lsp_outgoing_calls,
        'workflow-list': _handle_workflow_list,
        'workflow-get': _handle_workflow_get,
        'workflow-run': _handle_workflow_run,
        'trigger-list': _handle_trigger_list,
        'trigger-get': _handle_trigger_get,
        'trigger-create': _handle_trigger_create,
        'trigger-update': _handle_trigger_update,
        'trigger-run': _handle_trigger_run,
        'team-status': _handle_team_status,
        'team-list': _handle_team_list,
        'team-get': _handle_team_get,
        'team-create': _handle_team_create,
        'team-delete': _handle_team_delete,
        'team-messages': _handle_team_messages,
        'show-command': _handle_show_command,
        'show-tool': _handle_show_tool,
        'exec-command': _handle_exec_command,
        'exec-tool': _handle_exec_tool,
        'agent': _handle_agent,
        'agent-bg': _handle_agent_bg,
        'agent-bg-worker': _handle_agent_bg_worker,
        'agent-ps': _handle_agent_ps,
        'agent-logs': _handle_agent_logs,
        'agent-attach': _handle_agent_attach,
        'agent-kill': _handle_agent_kill,
        'daemon': _handle_daemon,
        'agent-chat': _handle_agent_chat,
        'agent-tui': _handle_agent_tui,
        'agent-resume': _handle_agent_resume,
        'agent-prompt': _handle_agent_prompt,
        'agent-context': _handle_agent_context,
        'agent-context-raw': _handle_agent_context_raw,
        'token-budget': _handle_token_budget,
        'agents': _handle_agents,
    }


def _handle_doctor(args: argparse.Namespace) -> int:
    resolved_api_key = _resolve_api_key(
        explicit_api_key=args.api_key,
        provider=_env_nonempty('LLM_PROVIDER'),
        model=args.model,
        base_url=args.base_url,
    )
    report = run_doctor(
        model_config=ModelConfig(
            model=args.model,
            base_url=args.base_url,
            api_key=resolved_api_key,
            timeout_seconds=args.timeout_seconds,
        ),
        runtime_config=AgentRuntimeConfig(cwd=_cwd(args)),
        check_backend=not args.skip_backend,
        check_tui=not args.skip_tui,
    )
    print(report.as_text())
    return 1 if report.has_failures else 0


def _handle_summary(args: argparse.Namespace) -> int:
    print(QueryEnginePort(build_port_manifest()).render_summary())
    return 0


def _handle_manifest(args: argparse.Namespace) -> int:
    print(build_port_manifest().to_markdown())
    return 0


def _handle_parity_audit(args: argparse.Namespace) -> int:
    print(run_parity_audit().to_markdown())
    return 0


def _handle_setup_report(args: argparse.Namespace) -> int:
    print(run_setup().as_markdown())
    return 0


def _handle_command_graph(args: argparse.Namespace) -> int:
    print(build_command_graph().as_markdown())
    return 0


def _handle_tool_pool(args: argparse.Namespace) -> int:
    print(assemble_tool_pool().as_markdown())
    return 0


def _handle_bootstrap_graph(args: argparse.Namespace) -> int:
    print(build_bootstrap_graph().as_markdown())
    return 0


def _handle_subsystems(args: argparse.Namespace) -> int:
    manifest = build_port_manifest()
    for subsystem in manifest.top_level_modules[: args.limit]:
        print(f'{subsystem.name}\t{subsystem.file_count}\t{subsystem.notes}')
    return 0


def _handle_commands(args: argparse.Namespace) -> int:
    if args.query:
        print(render_command_index(limit=args.limit, query=args.query))
        return 0
    commands = get_commands(
        include_plugin_commands=not args.no_plugin_commands,
        include_skill_commands=not args.no_skill_commands,
    )
    output_lines = [f'Command entries: {len(commands)}', '']
    output_lines.extend(
        f'- {module.name} â€” {module.source_hint}' for module in commands[: args.limit]
    )
    print('\n'.join(output_lines))
    return 0


def _handle_tools(args: argparse.Namespace) -> int:
    if args.query:
        print(render_tool_index(limit=args.limit, query=args.query))
        return 0
    permission_context = ToolPermissionContext.from_iterables(
        args.deny_tool,
        args.deny_prefix,
    )
    tools = get_tools(
        simple_mode=args.simple_mode,
        include_mcp=not args.no_mcp,
        permission_context=permission_context,
    )
    output_lines = [f'Tool entries: {len(tools)}', '']
    output_lines.extend(
        f'- {module.name} â€” {module.source_hint}' for module in tools[: args.limit]
    )
    print('\n'.join(output_lines))
    return 0


def _handle_route(args: argparse.Namespace) -> int:
    matches = PortRuntime().route_prompt(args.prompt, limit=args.limit)
    if not matches:
        print('No mirrored command/tool matches found.')
        return 0
    for match in matches:
        print(f'{match.kind}\t{match.name}\t{match.score}\t{match.source_hint}')
    return 0


def _handle_bootstrap(args: argparse.Namespace) -> int:
    print(PortRuntime().bootstrap_session(args.prompt, limit=args.limit).as_markdown())
    return 0


def _handle_turn_loop(args: argparse.Namespace) -> int:
    results = PortRuntime().run_turn_loop(
        args.prompt,
        limit=args.limit,
        max_turns=args.max_turns,
        structured_output=args.structured_output,
    )
    for idx, result in enumerate(results, start=1):
        print(f'## Turn {idx}')
        print(result.output)
        print(f'stop_reason={result.stop_reason}')
    return 0


def _handle_flush_transcript(args: argparse.Namespace) -> int:
    engine = QueryEnginePort.from_workspace()
    engine.submit_message(args.prompt)
    path = engine.persist_session()
    print(path)
    print(f'flushed={engine.transcript_store.flushed}')
    return 0


def _handle_load_session(args: argparse.Namespace) -> int:
    session = load_session(args.session_id)
    print(
        f'{session.session_id}\n{len(session.messages)} messages\n'
        f'in={session.input_tokens} out={session.output_tokens}'
    )
    return 0


def _handle_remote_mode(args: argparse.Namespace) -> int:
    print(run_remote_mode(args.target, cwd=_cwd(args)).as_text())
    return 0


def _handle_ssh_mode(args: argparse.Namespace) -> int:
    print(run_ssh_mode(args.target, cwd=_cwd(args)).as_text())
    return 0


def _handle_teleport_mode(args: argparse.Namespace) -> int:
    print(run_teleport_mode(args.target, cwd=_cwd(args)).as_text())
    return 0


def _handle_direct_connect_mode(args: argparse.Namespace) -> int:
    print(run_direct_connect_mode(args.target, cwd=_cwd(args)).as_text())
    return 0


def _handle_deep_link_mode(args: argparse.Namespace) -> int:
    print(run_deep_link_mode(args.target, cwd=_cwd(args)).as_text())
    return 0


def _handle_remote_status(args: argparse.Namespace) -> int:
    runtime = RemoteRuntime.from_workspace(_cwd(args))
    print('# Remote')
    print()
    print(runtime.render_summary())
    return 0


def _handle_remote_profiles(args: argparse.Namespace) -> int:
    runtime = RemoteRuntime.from_workspace(_cwd(args))
    print(runtime.render_profiles_index(query=args.query))
    return 0


def _handle_remote_disconnect(args: argparse.Namespace) -> int:
    runtime = RemoteRuntime.from_workspace(_cwd(args))
    print(runtime.disconnect().as_text())
    return 0


def _handle_worktree_status(args: argparse.Namespace) -> int:
    runtime = WorktreeRuntime.from_workspace(_cwd(args))
    print('# Worktree')
    print()
    print(runtime.render_summary())
    return 0


def _handle_worktree_enter(args: argparse.Namespace) -> int:
    runtime = WorktreeRuntime.from_workspace(_cwd(args))
    try:
        print(runtime.enter(name=args.name).as_text())
    except (RuntimeError, ValueError) as exc:
        print(exc)
        return 1
    return 0


def _handle_worktree_exit(args: argparse.Namespace) -> int:
    runtime = WorktreeRuntime.from_workspace(_cwd(args))
    try:
        print(
            runtime.exit(
                action=args.action,
                discard_changes=args.discard_changes,
            ).as_text()
        )
    except (RuntimeError, ValueError) as exc:
        print(exc)
        return 1
    return 0


def _handle_account_status(args: argparse.Namespace) -> int:
    runtime = AccountRuntime.from_workspace(_cwd(args))
    print('# Account')
    print()
    print(runtime.render_summary())
    return 0


def _handle_account_profiles(args: argparse.Namespace) -> int:
    runtime = AccountRuntime.from_workspace(_cwd(args))
    print(runtime.render_profiles_index(query=args.query))
    return 0


def _handle_account_login(args: argparse.Namespace) -> int:
    runtime = AccountRuntime.from_workspace(_cwd(args))
    print(
        runtime.login(
            args.target,
            provider=args.provider,
            auth_mode=args.auth_mode,
        ).as_text()
    )
    return 0


def _handle_account_logout(args: argparse.Namespace) -> int:
    runtime = AccountRuntime.from_workspace(_cwd(args))
    print(runtime.logout().as_text())
    return 0


def _handle_ask_status(args: argparse.Namespace) -> int:
    runtime = AskUserRuntime.from_workspace(_cwd(args))
    print('# Ask User')
    print()
    print(runtime.render_summary())
    return 0


def _handle_ask_history(args: argparse.Namespace) -> int:
    runtime = AskUserRuntime.from_workspace(_cwd(args))
    print(runtime.render_history())
    return 0


def _handle_search_status(args: argparse.Namespace) -> int:
    runtime = SearchRuntime.from_workspace(_cwd(args))
    if args.provider:
        print(runtime.render_provider(args.provider))
    else:
        print('# Search')
        print()
        print(runtime.render_summary())
    return 0


def _handle_search_providers(args: argparse.Namespace) -> int:
    runtime = SearchRuntime.from_workspace(_cwd(args))
    print(runtime.render_providers_index(query=args.query))
    return 0


def _handle_search_activate(args: argparse.Namespace) -> int:
    runtime = SearchRuntime.from_workspace(_cwd(args))
    try:
        report = runtime.activate_provider(args.provider)
    except KeyError:
        print(f'Unknown search provider: {args.provider}')
        return 1
    print(report.as_text())
    return 0


def _handle_search(args: argparse.Namespace) -> int:
    runtime = SearchRuntime.from_workspace(_cwd(args))
    try:
        output = runtime.render_search_results(
            args.query,
            provider_name=args.provider,
            max_results=args.max_results,
            domains=tuple(args.domain),
        )
    except (KeyError, LookupError, OSError, ValueError) as exc:
        print(f'Search failed: {exc}')
        return 1
    print(output)
    return 0


def _handle_mcp_status(args: argparse.Namespace) -> int:
    runtime = MCPRuntime.from_workspace(_cwd(args))
    print('# MCP')
    print()
    print(runtime.render_summary())
    return 0


def _handle_mcp_resources(args: argparse.Namespace) -> int:
    runtime = MCPRuntime.from_workspace(_cwd(args))
    print(runtime.render_resource_index(query=args.query))
    return 0


def _handle_mcp_resource(args: argparse.Namespace) -> int:
    runtime = MCPRuntime.from_workspace(_cwd(args))
    print(runtime.render_resource(args.uri))
    return 0


def _handle_mcp_tools(args: argparse.Namespace) -> int:
    runtime = MCPRuntime.from_workspace(_cwd(args))
    print(runtime.render_tool_index(query=args.query, server_name=args.server))
    return 0


def _handle_mcp_call_tool(args: argparse.Namespace) -> int:
    runtime = MCPRuntime.from_workspace(_cwd(args))
    arguments = json.loads(args.arguments_json)
    if not isinstance(arguments, dict):
        print('arguments-json must decode to a JSON object')
        return 1
    print(
        runtime.render_tool_call(
            args.tool_name,
            arguments=arguments,
            server_name=args.server,
        )
    )
    return 0


def _handle_config_status(args: argparse.Namespace) -> int:
    runtime = ConfigRuntime.from_workspace(_cwd(args))
    print('# Config')
    print()
    print(runtime.render_summary())
    return 0


def _handle_config_effective(args: argparse.Namespace) -> int:
    runtime = ConfigRuntime.from_workspace(_cwd(args))
    print(runtime.render_effective_config())
    return 0


def _handle_config_source(args: argparse.Namespace) -> int:
    runtime = ConfigRuntime.from_workspace(_cwd(args))
    print(runtime.render_source(args.source))
    return 0


def _handle_config_get(args: argparse.Namespace) -> int:
    runtime = ConfigRuntime.from_workspace(_cwd(args))
    print(runtime.render_value(args.key_path, source=args.source))
    return 0


def _handle_config_set(args: argparse.Namespace) -> int:
    runtime = ConfigRuntime.from_workspace(_cwd(args))
    value = json.loads(args.value_json)
    mutation = runtime.set_value(args.key_path, value, source=args.source)
    print('# Config')
    print()
    print(f'source={mutation.source_name}')
    print(f'key_path={mutation.key_path}')
    print(f'store_path={mutation.store_path}')
    print(f'effective_key_count={mutation.effective_key_count}')
    print(runtime.render_value(args.key_path))
    return 0


def _handle_lsp_status(args: argparse.Namespace) -> int:
    runtime = LSPRuntime.from_workspace(_cwd(args))
    print('# LSP')
    print()
    print(runtime.render_summary())
    return 0


def _handle_lsp_symbols(args: argparse.Namespace) -> int:
    runtime = LSPRuntime.from_workspace(_cwd(args))
    try:
        print(runtime.render_document_symbols(args.file_path))
    except KeyError as exc:
        print(exc)
        return 1
    return 0


def _handle_lsp_workspace_symbols(args: argparse.Namespace) -> int:
    runtime = LSPRuntime.from_workspace(_cwd(args))
    print(runtime.render_workspace_symbols(args.query, max_results=args.max_results))
    return 0


def _handle_lsp_definition(args: argparse.Namespace) -> int:
    runtime = LSPRuntime.from_workspace(_cwd(args))
    try:
        print(
            runtime.render_definition(
                args.file_path,
                args.line,
                args.character,
                max_results=args.max_results,
            )
        )
    except KeyError as exc:
        print(exc)
        return 1
    return 0


def _handle_lsp_references(args: argparse.Namespace) -> int:
    runtime = LSPRuntime.from_workspace(_cwd(args))
    try:
        print(
            runtime.render_references(
                args.file_path,
                args.line,
                args.character,
                max_results=args.max_results,
            )
        )
    except KeyError as exc:
        print(exc)
        return 1
    return 0


def _handle_lsp_hover(args: argparse.Namespace) -> int:
    runtime = LSPRuntime.from_workspace(_cwd(args))
    try:
        print(runtime.render_hover(args.file_path, args.line, args.character))
    except KeyError as exc:
        print(exc)
        return 1
    return 0


def _handle_lsp_diagnostics(args: argparse.Namespace) -> int:
    runtime = LSPRuntime.from_workspace(_cwd(args))
    try:
        print(runtime.render_diagnostics(args.file_path))
    except KeyError as exc:
        print(exc)
        return 1
    return 0


def _handle_lsp_call_hierarchy(args: argparse.Namespace) -> int:
    runtime = LSPRuntime.from_workspace(_cwd(args))
    try:
        print(
            runtime.render_prepare_call_hierarchy(
                args.file_path,
                args.line,
                args.character,
            )
        )
    except KeyError as exc:
        print(exc)
        return 1
    return 0


def _handle_lsp_incoming_calls(args: argparse.Namespace) -> int:
    runtime = LSPRuntime.from_workspace(_cwd(args))
    try:
        print(
            runtime.render_incoming_calls(
                args.file_path,
                args.line,
                args.character,
                max_results=args.max_results,
            )
        )
    except KeyError as exc:
        print(exc)
        return 1
    return 0


def _handle_lsp_outgoing_calls(args: argparse.Namespace) -> int:
    runtime = LSPRuntime.from_workspace(_cwd(args))
    try:
        print(
            runtime.render_outgoing_calls(
                args.file_path,
                args.line,
                args.character,
                max_results=args.max_results,
            )
        )
    except KeyError as exc:
        print(exc)
        return 1
    return 0


def _handle_workflow_list(args: argparse.Namespace) -> int:
    runtime = WorkflowRuntime.from_workspace(_cwd(args))
    print(runtime.render_workflows_index(query=args.query))
    return 0


def _handle_workflow_get(args: argparse.Namespace) -> int:
    runtime = WorkflowRuntime.from_workspace(_cwd(args))
    try:
        print(runtime.render_workflow(args.workflow_name))
    except KeyError:
        print(f'Unknown workflow: {args.workflow_name}')
        return 1
    return 0


def _handle_workflow_run(args: argparse.Namespace) -> int:
    runtime = WorkflowRuntime.from_workspace(_cwd(args))
    arguments = json.loads(args.arguments_json)
    if not isinstance(arguments, dict):
        print('arguments-json must decode to a JSON object')
        return 1
    try:
        print(runtime.render_run_report(args.workflow_name, arguments=arguments))
    except KeyError:
        print(f'Unknown workflow: {args.workflow_name}')
        return 1
    return 0


def _handle_trigger_list(args: argparse.Namespace) -> int:
    runtime = RemoteTriggerRuntime.from_workspace(_cwd(args))
    print(runtime.render_trigger_index(query=args.query))
    return 0


def _handle_trigger_get(args: argparse.Namespace) -> int:
    runtime = RemoteTriggerRuntime.from_workspace(_cwd(args))
    try:
        print(runtime.render_trigger(args.trigger_id))
    except KeyError:
        print(f'Unknown remote trigger: {args.trigger_id}')
        return 1
    return 0


def _handle_trigger_create(args: argparse.Namespace) -> int:
    runtime = RemoteTriggerRuntime.from_workspace(_cwd(args))
    body = json.loads(args.body_json)
    if not isinstance(body, dict):
        print('body-json must decode to a JSON object')
        return 1
    try:
        trigger = runtime.create_trigger(body)
    except (KeyError, TypeError, ValueError) as exc:
        print(exc)
        return 1
    print(runtime.render_trigger(trigger.trigger_id))
    return 0


def _handle_trigger_update(args: argparse.Namespace) -> int:
    runtime = RemoteTriggerRuntime.from_workspace(_cwd(args))
    body = json.loads(args.body_json)
    if not isinstance(body, dict):
        print('body-json must decode to a JSON object')
        return 1
    try:
        trigger = runtime.update_trigger(args.trigger_id, body)
    except (KeyError, TypeError, ValueError) as exc:
        print(exc)
        return 1
    print(runtime.render_trigger(trigger.trigger_id))
    return 0


def _handle_trigger_run(args: argparse.Namespace) -> int:
    runtime = RemoteTriggerRuntime.from_workspace(_cwd(args))
    body = json.loads(args.body_json)
    if not isinstance(body, dict):
        print('body-json must decode to a JSON object')
        return 1
    try:
        print(runtime.render_run_report(args.trigger_id, body=body))
    except KeyError:
        print(f'Unknown remote trigger: {args.trigger_id}')
        return 1
    return 0


def _handle_team_status(args: argparse.Namespace) -> int:
    runtime = TeamRuntime.from_workspace(_cwd(args))
    print('# Teams')
    print()
    print(runtime.render_summary())
    return 0


def _handle_team_list(args: argparse.Namespace) -> int:
    runtime = TeamRuntime.from_workspace(_cwd(args))
    print(runtime.render_teams_index(query=args.query))
    return 0


def _handle_team_get(args: argparse.Namespace) -> int:
    runtime = TeamRuntime.from_workspace(_cwd(args))
    try:
        print(runtime.render_team(args.team_name))
    except KeyError:
        print(f'Unknown team: {args.team_name}')
        return 1
    return 0


def _handle_team_create(args: argparse.Namespace) -> int:
    runtime = TeamRuntime.from_workspace(_cwd(args))
    try:
        team = runtime.create_team(
            args.team_name,
            description=args.description,
            members=args.member,
        )
    except KeyError:
        print(f'Team already exists: {args.team_name}')
        return 1
    print(f'created team {team.name}')
    return 0


def _handle_team_delete(args: argparse.Namespace) -> int:
    runtime = TeamRuntime.from_workspace(_cwd(args))
    try:
        team = runtime.delete_team(args.team_name)
    except KeyError:
        print(f'Unknown team: {args.team_name}')
        return 1
    print(f'deleted team {team.name}')
    return 0


def _handle_team_messages(args: argparse.Namespace) -> int:
    runtime = TeamRuntime.from_workspace(_cwd(args))
    try:
        print(runtime.render_messages(team_name=args.team_name))
    except KeyError:
        print(f'Unknown team: {args.team_name}')
        return 1
    return 0


def _handle_show_command(args: argparse.Namespace) -> int:
    module = get_command(args.name)
    if module is None:
        print(f'Command not found: {args.name}')
        return 1
    print('\n'.join([module.name, module.source_hint, module.responsibility]))
    return 0


def _handle_show_tool(args: argparse.Namespace) -> int:
    module = get_tool(args.name)
    if module is None:
        print(f'Tool not found: {args.name}')
        return 1
    print('\n'.join([module.name, module.source_hint, module.responsibility]))
    return 0


def _handle_exec_command(args: argparse.Namespace) -> int:
    result = execute_command(args.name, args.prompt)
    print(result.message)
    return 0 if result.handled else 1


def _handle_exec_tool(args: argparse.Namespace) -> int:
    result = execute_tool(args.name, args.payload)
    print(result.message)
    return 0 if result.handled else 1


def _handle_agent(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    _run_agent_turn(
        agent,
        args.prompt,
        show_transcript=args.show_transcript,
        show_usage=getattr(args, 'show_usage', True),
    )
    return 0


def _handle_agent_bg(args: argparse.Namespace) -> int:
    return _launch_background_agent(args)


def _handle_agent_bg_worker(args: argparse.Namespace) -> int:
    return _run_background_worker(args)


def _handle_agent_ps(args: argparse.Namespace) -> int:
    print(BackgroundSessionRuntime().render_ps())
    return 0


def _handle_agent_logs(args: argparse.Namespace) -> int:
    print(BackgroundSessionRuntime().render_logs(args.background_id, tail=args.tail))
    return 0


def _handle_agent_attach(args: argparse.Namespace) -> int:
    print(BackgroundSessionRuntime().render_attach(args.background_id, tail=args.tail))
    return 0


def _handle_agent_kill(args: argparse.Namespace) -> int:
    record = BackgroundSessionRuntime().kill(args.background_id)
    print('# Background Session')
    print(f'background_id={record.background_id}')
    print(f'status={record.status}')
    print(f'pid={record.pid}')
    if record.exit_code is not None:
        print(f'exit_code={record.exit_code}')
    return 0


def _handle_daemon(args: argparse.Namespace) -> int:
    daemon_handlers: dict[str, CommandHandler] = {
        'start': _handle_agent_bg,
        'worker': _handle_agent_bg_worker,
        'ps': _handle_agent_ps,
        'logs': _handle_agent_logs,
        'attach': _handle_agent_attach,
        'kill': _handle_agent_kill,
    }
    handler = daemon_handlers.get(str(args.daemon_command))
    if handler is None:
        raise ValueError(f'Unknown daemon subcommand: {args.daemon_command}')
    return handler(args)


def _handle_agent_chat(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    return _run_agent_chat_loop(
        agent,
        initial_prompt=args.prompt,
        resume_session_id=args.resume_session_id,
        show_transcript=args.show_transcript,
        show_usage=getattr(args, 'show_usage', False),
    )


def _handle_agent_tui(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    try:
        return run_agent_tui(
            agent,
            initial_prompt=args.prompt,
            resume_session_id=args.resume_session_id,
        )
    except RuntimeError as exc:
        print(exc)
        return 1


def _handle_agent_resume(args: argparse.Namespace) -> int:
    agent, stored_session = _build_resumed_agent(args)
    _run_agent_turn(
        agent,
        args.prompt,
        show_transcript=args.show_transcript,
        show_usage=getattr(args, 'show_usage', True),
        stored_session=stored_session,
    )
    return 0


def _handle_agent_prompt(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    print(agent.render_system_prompt())
    return 0


def _handle_agent_context(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    print(agent.render_context_report())
    return 0


def _handle_agent_context_raw(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    print(agent.render_context_snapshot_report())
    return 0


def _handle_token_budget(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    agent.last_session = agent.build_session()
    print(agent.render_token_budget_report())
    return 0


def _handle_agents(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    if args.agent_type:
        print(agent.render_agent_detail_report(args.agent_type))
    else:
        print(agent.render_agents_report(show_all=bool(args.all)))
    return 0
