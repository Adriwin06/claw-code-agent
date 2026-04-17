from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.features.account_runtime import AccountRuntime
from src.features.ask_user_runtime import AskUserRuntime
from src.agent.agent_types import AgentRuntimeConfig, ModelConfig
from src.features.background_runtime import BackgroundSessionRuntime
from src.core.bootstrap_runtime import build_bootstrap_graph, run_setup
from src.core.catalog_runtime import (
    build_command_graph,
    build_port_manifest,
    assemble_tool_pool,
    execute_command,
    execute_tool,
    get_command,
    get_commands,
    get_tool,
    get_tools,
    render_command_index,
    render_tool_index,
)
from src.features.config_runtime import ConfigRuntime
from src.features.doctor_runtime import run_doctor
from src.features.lsp_runtime import LSPRuntime
from src.features.mcp_runtime import MCPRuntime
from src.core.parity_audit import run_parity_audit
from src.core.permissions import ToolPermissionContext
from src.core.query_engine import QueryEnginePort
from src.features.remote_runtime import (
    RemoteRuntime,
    run_deep_link_mode,
    run_direct_connect_mode,
    run_remote_mode,
    run_ssh_mode,
    run_teleport_mode,
)
from src.features.remote_trigger_runtime import RemoteTriggerRuntime
from src.core.runtime import PortRuntime
from src.features.search_runtime import SearchRuntime
from src.session.session_store import load_session
from src.features.task_runtime import TaskRuntime
from src.features.team_runtime import TeamRuntime
from ..textual_ui import run_agent_tui
from src.features.workflow_runtime import WorkflowRuntime
from src.features.worktree_runtime import WorktreeRuntime
from .agent_cli_config import _build_agent
from .agent_runtime_ops import (
    _build_resumed_agent,
    _launch_background_agent,
    _run_agent_chat_loop,
    _run_agent_turn,
    _run_background_worker,
)


def dispatch_main_command(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
) -> int:
    if args.command == 'dev':
        args.command = args.dev_command
    manifest = build_port_manifest()

    if args.command == 'doctor':
        report = run_doctor(
            model_config=ModelConfig(
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                timeout_seconds=args.timeout_seconds,
            ),
            runtime_config=AgentRuntimeConfig(cwd=Path(args.cwd).resolve()),
            check_backend=not args.skip_backend,
            check_tui=not args.skip_tui,
        )
        print(report.as_text())
        return 1 if report.has_failures else 0

    if args.command == 'summary':
        print(QueryEnginePort(manifest).render_summary())
        return 0
    if args.command == 'manifest':
        print(manifest.to_markdown())
        return 0
    if args.command == 'parity-audit':
        print(run_parity_audit().to_markdown())
        return 0
    if args.command == 'setup-report':
        print(run_setup().as_markdown())
        return 0
    if args.command == 'command-graph':
        print(build_command_graph().as_markdown())
        return 0
    if args.command == 'tool-pool':
        print(assemble_tool_pool().as_markdown())
        return 0
    if args.command == 'bootstrap-graph':
        print(build_bootstrap_graph().as_markdown())
        return 0
    if args.command == 'subsystems':
        for subsystem in manifest.top_level_modules[: args.limit]:
            print(f'{subsystem.name}\t{subsystem.file_count}\t{subsystem.notes}')
        return 0
    if args.command == 'commands':
        if args.query:
            print(render_command_index(limit=args.limit, query=args.query))
        else:
            commands = get_commands(
                include_plugin_commands=not args.no_plugin_commands,
                include_skill_commands=not args.no_skill_commands,
            )
            output_lines = [f'Command entries: {len(commands)}', '']
            output_lines.extend(f'- {module.name} — {module.source_hint}' for module in commands[: args.limit])
            print('\n'.join(output_lines))
        return 0
    if args.command == 'tools':
        if args.query:
            print(render_tool_index(limit=args.limit, query=args.query))
        else:
            permission_context = ToolPermissionContext.from_iterables(args.deny_tool, args.deny_prefix)
            tools = get_tools(
                simple_mode=args.simple_mode,
                include_mcp=not args.no_mcp,
                permission_context=permission_context,
            )
            output_lines = [f'Tool entries: {len(tools)}', '']
            output_lines.extend(f'- {module.name} — {module.source_hint}' for module in tools[: args.limit])
            print('\n'.join(output_lines))
        return 0
    if args.command == 'route':
        matches = PortRuntime().route_prompt(args.prompt, limit=args.limit)
        if not matches:
            print('No mirrored command/tool matches found.')
            return 0
        for match in matches:
            print(f'{match.kind}\t{match.name}\t{match.score}\t{match.source_hint}')
        return 0
    if args.command == 'bootstrap':
        print(PortRuntime().bootstrap_session(args.prompt, limit=args.limit).as_markdown())
        return 0
    if args.command == 'turn-loop':
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
    if args.command == 'flush-transcript':
        engine = QueryEnginePort.from_workspace()
        engine.submit_message(args.prompt)
        path = engine.persist_session()
        print(path)
        print(f'flushed={engine.transcript_store.flushed}')
        return 0
    if args.command == 'load-session':
        session = load_session(args.session_id)
        print(f'{session.session_id}\n{len(session.messages)} messages\nin={session.input_tokens} out={session.output_tokens}')
        return 0
    if args.command == 'remote-mode':
        print(run_remote_mode(args.target, cwd=Path(args.cwd).resolve()).as_text())
        return 0
    if args.command == 'ssh-mode':
        print(run_ssh_mode(args.target, cwd=Path(args.cwd).resolve()).as_text())
        return 0
    if args.command == 'teleport-mode':
        print(run_teleport_mode(args.target, cwd=Path(args.cwd).resolve()).as_text())
        return 0
    if args.command == 'direct-connect-mode':
        print(run_direct_connect_mode(args.target, cwd=Path(args.cwd).resolve()).as_text())
        return 0
    if args.command == 'deep-link-mode':
        print(run_deep_link_mode(args.target, cwd=Path(args.cwd).resolve()).as_text())
        return 0
    if args.command == 'remote-status':
        runtime = RemoteRuntime.from_workspace(Path(args.cwd).resolve())
        print('# Remote')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'remote-profiles':
        runtime = RemoteRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_profiles_index(query=args.query))
        return 0
    if args.command == 'remote-disconnect':
        runtime = RemoteRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.disconnect().as_text())
        return 0
    if args.command == 'worktree-status':
        runtime = WorktreeRuntime.from_workspace(Path(args.cwd).resolve())
        print('# Worktree')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'worktree-enter':
        runtime = WorktreeRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.enter(name=args.name).as_text())
        except (RuntimeError, ValueError) as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'worktree-exit':
        runtime = WorktreeRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'account-status':
        runtime = AccountRuntime.from_workspace(Path(args.cwd).resolve())
        print('# Account')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'account-profiles':
        runtime = AccountRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_profiles_index(query=args.query))
        return 0
    if args.command == 'account-login':
        runtime = AccountRuntime.from_workspace(Path(args.cwd).resolve())
        print(
            runtime.login(
                args.target,
                provider=args.provider,
                auth_mode=args.auth_mode,
            ).as_text()
        )
        return 0
    if args.command == 'account-logout':
        runtime = AccountRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.logout().as_text())
        return 0
    if args.command == 'ask-status':
        runtime = AskUserRuntime.from_workspace(Path(args.cwd).resolve())
        print('# Ask User')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'ask-history':
        runtime = AskUserRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_history())
        return 0
    if args.command == 'search-status':
        runtime = SearchRuntime.from_workspace(Path(args.cwd).resolve())
        if args.provider:
            print(runtime.render_provider(args.provider))
        else:
            print('# Search')
            print()
            print(runtime.render_summary())
        return 0
    if args.command == 'search-providers':
        runtime = SearchRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_providers_index(query=args.query))
        return 0
    if args.command == 'search-activate':
        runtime = SearchRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            report = runtime.activate_provider(args.provider)
        except KeyError:
            print(f'Unknown search provider: {args.provider}')
            return 1
        print(report.as_text())
        return 0
    if args.command == 'search':
        runtime = SearchRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'mcp-status':
        runtime = MCPRuntime.from_workspace(Path(args.cwd).resolve())
        print('# MCP')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'mcp-resources':
        runtime = MCPRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_resource_index(query=args.query))
        return 0
    if args.command == 'mcp-resource':
        runtime = MCPRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_resource(args.uri))
        return 0
    if args.command == 'mcp-tools':
        runtime = MCPRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_tool_index(query=args.query, server_name=args.server))
        return 0
    if args.command == 'mcp-call-tool':
        runtime = MCPRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'config-status':
        runtime = ConfigRuntime.from_workspace(Path(args.cwd).resolve())
        print('# Config')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'config-effective':
        runtime = ConfigRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_effective_config())
        return 0
    if args.command == 'config-source':
        runtime = ConfigRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_source(args.source))
        return 0
    if args.command == 'config-get':
        runtime = ConfigRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_value(args.key_path, source=args.source))
        return 0
    if args.command == 'config-set':
        runtime = ConfigRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'lsp-status':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        print('# LSP')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'lsp-symbols':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_document_symbols(args.file_path))
        except KeyError as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'lsp-workspace-symbols':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_workspace_symbols(args.query, max_results=args.max_results))
        return 0
    if args.command == 'lsp-definition':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'lsp-references':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'lsp-hover':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_hover(args.file_path, args.line, args.character))
        except KeyError as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'lsp-diagnostics':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_diagnostics(args.file_path))
        except KeyError as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'lsp-call-hierarchy':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'lsp-incoming-calls':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'lsp-outgoing-calls':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'workflow-list':
        runtime = WorkflowRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_workflows_index(query=args.query))
        return 0
    if args.command == 'workflow-get':
        runtime = WorkflowRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_workflow(args.workflow_name))
        except KeyError:
            print(f'Unknown workflow: {args.workflow_name}')
            return 1
        return 0
    if args.command == 'workflow-run':
        runtime = WorkflowRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'trigger-list':
        runtime = RemoteTriggerRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_trigger_index(query=args.query))
        return 0
    if args.command == 'trigger-get':
        runtime = RemoteTriggerRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_trigger(args.trigger_id))
        except KeyError:
            print(f'Unknown remote trigger: {args.trigger_id}')
            return 1
        return 0
    if args.command == 'trigger-create':
        runtime = RemoteTriggerRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'trigger-update':
        runtime = RemoteTriggerRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'trigger-run':
        runtime = RemoteTriggerRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'team-status':
        runtime = TeamRuntime.from_workspace(Path(args.cwd).resolve())
        print('# Teams')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'team-list':
        runtime = TeamRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_teams_index(query=args.query))
        return 0
    if args.command == 'team-get':
        runtime = TeamRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_team(args.team_name))
        except KeyError:
            print(f'Unknown team: {args.team_name}')
            return 1
        return 0
    if args.command == 'team-create':
        runtime = TeamRuntime.from_workspace(Path(args.cwd).resolve())
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
    if args.command == 'team-delete':
        runtime = TeamRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            team = runtime.delete_team(args.team_name)
        except KeyError:
            print(f'Unknown team: {args.team_name}')
            return 1
        print(f'deleted team {team.name}')
        return 0
    if args.command == 'team-messages':
        runtime = TeamRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_messages(team_name=args.team_name))
        except KeyError:
            print(f'Unknown team: {args.team_name}')
            return 1
        return 0
    if args.command == 'show-command':
        module = get_command(args.name)
        if module is None:
            print(f'Command not found: {args.name}')
            return 1
        print('\n'.join([module.name, module.source_hint, module.responsibility]))
        return 0
    if args.command == 'show-tool':
        module = get_tool(args.name)
        if module is None:
            print(f'Tool not found: {args.name}')
            return 1
        print('\n'.join([module.name, module.source_hint, module.responsibility]))
        return 0
    if args.command == 'exec-command':
        result = execute_command(args.name, args.prompt)
        print(result.message)
        return 0 if result.handled else 1
    if args.command == 'exec-tool':
        result = execute_tool(args.name, args.payload)
        print(result.message)
        return 0 if result.handled else 1
    if args.command == 'agent':
        agent = _build_agent(args)
        _run_agent_turn(
            agent,
            args.prompt,
            show_transcript=args.show_transcript,
        )
        return 0
    if args.command == 'agent-bg':
        return _launch_background_agent(args)
    if args.command == 'agent-bg-worker':
        return _run_background_worker(args)
    if args.command == 'agent-ps':
        print(BackgroundSessionRuntime().render_ps())
        return 0
    if args.command == 'agent-logs':
        print(
            BackgroundSessionRuntime().render_logs(
                args.background_id,
                tail=args.tail,
            )
        )
        return 0
    if args.command == 'agent-attach':
        print(
            BackgroundSessionRuntime().render_attach(
                args.background_id,
                tail=args.tail,
            )
        )
        return 0
    if args.command == 'agent-kill':
        record = BackgroundSessionRuntime().kill(args.background_id)
        print('# Background Session')
        print(f'background_id={record.background_id}')
        print(f'status={record.status}')
        print(f'pid={record.pid}')
        if record.exit_code is not None:
            print(f'exit_code={record.exit_code}')
        return 0
    if args.command == 'daemon':
        if args.daemon_command == 'start':
            return _launch_background_agent(args)
        if args.daemon_command == 'worker':
            return _run_background_worker(args)
        if args.daemon_command == 'ps':
            print(BackgroundSessionRuntime().render_ps())
            return 0
        if args.daemon_command == 'logs':
            print(
                BackgroundSessionRuntime().render_logs(
                    args.background_id,
                    tail=args.tail,
                )
            )
            return 0
        if args.daemon_command == 'attach':
            print(
                BackgroundSessionRuntime().render_attach(
                    args.background_id,
                    tail=args.tail,
                )
            )
            return 0
        if args.daemon_command == 'kill':
            record = BackgroundSessionRuntime().kill(args.background_id)
            print('# Background Session')
            print(f'background_id={record.background_id}')
            print(f'status={record.status}')
            print(f'pid={record.pid}')
            if record.exit_code is not None:
                print(f'exit_code={record.exit_code}')
            return 0
    if args.command == 'agent-chat':
        agent = _build_agent(args)
        return _run_agent_chat_loop(
            agent,
            initial_prompt=args.prompt,
            resume_session_id=args.resume_session_id,
            show_transcript=args.show_transcript,
        )
    if args.command == 'agent-tui':
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
    if args.command == 'agent-resume':
        agent, stored_session = _build_resumed_agent(args)
        _run_agent_turn(
            agent,
            args.prompt,
            show_transcript=args.show_transcript,
            stored_session=stored_session,
        )
        return 0
    if args.command == 'agent-prompt':
        agent = _build_agent(args)
        print(agent.render_system_prompt())
        return 0
    if args.command == 'agent-context':
        agent = _build_agent(args)
        print(agent.render_context_report())
        return 0
    if args.command == 'agent-context-raw':
        agent = _build_agent(args)
        print(agent.render_context_snapshot_report())
        return 0
    if args.command == 'token-budget':
        agent = _build_agent(args)
        agent.last_session = agent.build_session()
        print(agent.render_token_budget_report())
        return 0
    if args.command == 'agents':
        agent = _build_agent(args)
        if args.agent_type:
            print(agent.render_agent_detail_report(args.agent_type))
        else:
            print(agent.render_agents_report(show_all=bool(args.all)))
        return 0

    parser.error(f'unknown command: {args.command}')
    return 2
