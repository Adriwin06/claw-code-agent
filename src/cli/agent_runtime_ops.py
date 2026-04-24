from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Callable, TextIO

from src.agent.runtime.agent import LocalCodingAgent
from src.agent.models.types import (
    AgentPermissions,
    BudgetConfig,
    ModelPricing,
)
from src.features.system.background_runtime import BackgroundSessionRuntime, build_background_worker_command
from ..cli.agent_cli_config import (
    _append_agent_forwarded_args,
    _build_agent,
    _load_output_schema_config,
)
from src.session.session_store import (
    StoredAgentSession,
    deserialize_model_config,
    deserialize_runtime_config,
    load_agent_session,
)


def _launch_background_agent(args) -> int:
    background_runtime = BackgroundSessionRuntime()
    background_id = background_runtime.create_id()
    forwarded_args: list[str] = []
    _append_agent_forwarded_args(forwarded_args, args, include_backend=True)
    forwarded_args.extend(['--background-root', str(background_runtime.root)])
    command = build_background_worker_command(
        background_id=background_id,
        prompt=args.prompt,
        forwarded_args=forwarded_args,
    )
    record = background_runtime.launch(
        command,
        prompt=args.prompt,
        workspace_cwd=Path(args.cwd).resolve(),
        model=args.model,
        background_id=background_id,
        process_cwd=Path(__file__).resolve().parent.parent.parent,
    )
    print('# Background Session')
    print(f'background_id={record.background_id}')
    print(f'pid={record.pid}')
    print(f'log_path={record.log_path}')
    print(f'record_path={record.record_path}')
    return 0


def _run_background_worker(args) -> int:
    background_runtime = BackgroundSessionRuntime(Path(args.background_root))
    exit_code = 1
    stop_reason = 'worker_failed'
    session_id = None
    session_path = None
    try:
        agent = _build_agent(args)
        result = _run_agent_turn(
            agent,
            args.prompt,
            show_transcript=args.show_transcript,
        )
        exit_code = 0
        stop_reason = result.stop_reason or 'completed'
        session_id = result.session_id
        session_path = result.session_path
        return 0
    finally:
        background_runtime.mark_finished(
            args.background_id,
            exit_code=exit_code,
            stop_reason=stop_reason,
            session_id=session_id,
            session_path=session_path,
        )


def _build_resumed_agent(args) -> tuple[LocalCodingAgent, StoredAgentSession]:
    stored_session = load_agent_session(args.session_id)
    model_config = deserialize_model_config(stored_session.model_config)
    runtime_config = deserialize_runtime_config(stored_session.runtime_config)

    if args.model:
        model_config = replace(model_config, model=args.model)
    if args.base_url:
        model_config = replace(model_config, base_url=args.base_url)
    if args.api_key:
        model_config = replace(model_config, api_key=args.api_key)
    if args.llm_backend:
        model_config = replace(model_config, llm_backend=args.llm_backend)
    if args.temperature is not None:
        model_config = replace(model_config, temperature=args.temperature)
    if args.timeout_seconds is not None:
        model_config = replace(model_config, timeout_seconds=args.timeout_seconds)
    if args.input_cost_per_million is not None or args.output_cost_per_million is not None:
        model_config = replace(
            model_config,
            pricing=replace(
                model_config.pricing,
                input_cost_per_million_tokens_usd=(
                    args.input_cost_per_million
                    if args.input_cost_per_million is not None
                    else model_config.pricing.input_cost_per_million_tokens_usd
                ),
                output_cost_per_million_tokens_usd=(
                    args.output_cost_per_million
                    if args.output_cost_per_million is not None
                    else model_config.pricing.output_cost_per_million_tokens_usd
                ),
            ),
        )

    if args.max_turns is not None:
        runtime_config = replace(runtime_config, max_turns=args.max_turns)
    if args.allow_write or args.allow_shell or args.unsafe:
        runtime_config = replace(
            runtime_config,
            permissions=AgentPermissions(
                allow_file_write=runtime_config.permissions.allow_file_write or args.allow_write,
                allow_shell_commands=runtime_config.permissions.allow_shell_commands or args.allow_shell,
                allow_destructive_shell_commands=runtime_config.permissions.allow_destructive_shell_commands or args.unsafe,
            ),
        )
    if args.stream:
        runtime_config = replace(runtime_config, stream_model_responses=True)
    if (
        args.auto_snip_threshold is not None
        or args.auto_compact_threshold is not None
        or args.compact_preserve_messages is not None
    ):
        runtime_config = replace(
            runtime_config,
            auto_snip_threshold_tokens=(
                args.auto_snip_threshold
                if args.auto_snip_threshold is not None
                else runtime_config.auto_snip_threshold_tokens
            ),
            auto_compact_threshold_tokens=(
                args.auto_compact_threshold
                if args.auto_compact_threshold is not None
                else runtime_config.auto_compact_threshold_tokens
            ),
            compact_preserve_messages=(
                max(0, args.compact_preserve_messages)
                if args.compact_preserve_messages is not None
                else runtime_config.compact_preserve_messages
            ),
        )
    if (
        args.max_total_tokens is not None
        or args.max_input_tokens is not None
        or args.max_output_tokens is not None
        or args.max_reasoning_tokens is not None
        or args.max_budget_usd is not None
        or args.max_tool_calls is not None
        or args.max_delegated_tasks is not None
        or args.max_model_calls is not None
        or args.max_session_turns is not None
    ):
        runtime_config = replace(
            runtime_config,
            budget_config=BudgetConfig(
                max_total_tokens=(
                    args.max_total_tokens
                    if args.max_total_tokens is not None
                    else runtime_config.budget_config.max_total_tokens
                ),
                max_input_tokens=(
                    args.max_input_tokens
                    if args.max_input_tokens is not None
                    else runtime_config.budget_config.max_input_tokens
                ),
                max_output_tokens=(
                    args.max_output_tokens
                    if args.max_output_tokens is not None
                    else runtime_config.budget_config.max_output_tokens
                ),
                max_reasoning_tokens=(
                    args.max_reasoning_tokens
                    if args.max_reasoning_tokens is not None
                    else runtime_config.budget_config.max_reasoning_tokens
                ),
                max_total_cost_usd=(
                    args.max_budget_usd
                    if args.max_budget_usd is not None
                    else runtime_config.budget_config.max_total_cost_usd
                ),
                max_tool_calls=(
                    args.max_tool_calls
                    if args.max_tool_calls is not None
                    else runtime_config.budget_config.max_tool_calls
                ),
                max_delegated_tasks=(
                    args.max_delegated_tasks
                    if args.max_delegated_tasks is not None
                    else runtime_config.budget_config.max_delegated_tasks
                ),
                max_model_calls=(
                    args.max_model_calls
                    if args.max_model_calls is not None
                    else runtime_config.budget_config.max_model_calls
                ),
                max_session_turns=(
                    args.max_session_turns
                    if args.max_session_turns is not None
                    else runtime_config.budget_config.max_session_turns
                ),
            ),
        )
    output_schema = _load_output_schema_config(args)
    if output_schema is not None:
        runtime_config = replace(runtime_config, output_schema=output_schema)
    if args.scratchpad_root:
        runtime_config = replace(
            runtime_config,
            scratchpad_root=Path(args.scratchpad_root).resolve(),
        )

    agent = LocalCodingAgent(
        model_config=model_config,
        runtime_config=runtime_config,
        llm_backend=model_config.llm_backend,
    )
    return agent, stored_session


def _preview_value(value: object, *, max_chars: int = 160) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=True, sort_keys=True)
        except (TypeError, ValueError):
            text = str(value)
    text = ' '.join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 3] + '...'
    return text


class _AgentLiveRenderer:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self.streamed_assistant_output = False
        self._assistant_open = False
        self._assistant_ends_with_newline = True
        self._tool_stream_key: tuple[str | None, str | None] | None = None
        self._tool_stream_open = False
        self._tool_stream_ends_with_newline = True
        self._announced_tool_plans: set[str] = set()

    def handle_event(self, event: dict[str, object]) -> None:
        event_type = event.get('type')
        if event_type == 'message_start':
            self._write_status('[thinking] model call started')
            return
        if event_type == 'content_delta':
            self._render_assistant_delta(str(event.get('delta', '')))
            return
        if event_type == 'tool_call_delta':
            self._render_tool_plan(event)
            return
        if event_type == 'message_stop':
            finish_reason = event.get('finish_reason')
            if finish_reason == 'tool_calls':
                self._write_status('[thinking] dispatching tool call')
            elif finish_reason == 'length':
                self._write_status('[thinking] model output hit the length limit')
            else:
                self._close_open_blocks()
            return
        if event_type == 'tool_start':
            self._write_status(self._render_tool_start(event))
            return
        if event_type == 'tool_delta':
            self._render_tool_delta(event)
            return
        if event_type == 'tool_result':
            rendered_result = self._render_tool_result(event)
            self._write_status(rendered_result)
            if not bool(event.get('ok')):
                detail = _preview_value(event.get('content'), max_chars=1200)
                if detail:
                    self._write_status(f'[tool-error] {detail}')
            return
        if event_type == 'usage':
            usage = event.get('usage')
            if isinstance(usage, dict):
                parts = [
                    f"input={usage.get('input_tokens', 0)}",
                    f"output={usage.get('output_tokens', 0)}",
                ]
                reasoning = usage.get('reasoning_tokens', 0)
                if reasoning:
                    parts.append(f'reasoning={reasoning}')
                self._write_status('[usage] ' + ' '.join(parts))
            return
        if event_type == 'continuation_request':
            self._write_status('[thinking] requesting continuation')
            return
        if event_type == 'tool_permission_denial':
            reason = _preview_value(event.get('reason'), max_chars=220)
            self._write_status(f'[tool] permission denied: {reason}')
            return
        if event_type == 'task_budget_exceeded':
            reason = _preview_value(event.get('reason'), max_chars=220)
            self._write_status(f'[budget] {reason}')
            return
        if event_type == 'plugin_tool_block':
            message = _preview_value(event.get('message'), max_chars=220)
            self._write_status(f'[plugin] blocked tool: {message}')
            return
        if event_type == 'hook_policy_tool_block':
            message = _preview_value(event.get('message'), max_chars=220)
            self._write_status(f'[policy] blocked tool: {message}')
            return

    def finish(self) -> None:
        self._close_open_blocks()

    def _render_assistant_delta(self, delta: str) -> None:
        if not delta:
            return
        self._close_tool_stream()
        if not self._assistant_open:
            self._write('[assistant] ')
            self._assistant_open = True
            self._assistant_ends_with_newline = False
        self._write(delta)
        self.streamed_assistant_output = True
        self._assistant_ends_with_newline = delta.endswith('\n')

    def _render_tool_plan(self, event: dict[str, object]) -> None:
        tool_call_id = event.get('tool_call_id')
        tool_name = event.get('tool_name')
        if not isinstance(tool_name, str) or not tool_name:
            return
        key = str(tool_call_id or f"index:{event.get('tool_call_index', 0)}")
        if key in self._announced_tool_plans:
            return
        self._announced_tool_plans.add(key)
        arguments_delta = _preview_value(event.get('arguments_delta'), max_chars=360)
        plan_content = f'Planning tool call: {tool_name}'
        if isinstance(tool_call_id, str) and tool_call_id:
            plan_content += f' id={tool_call_id}'
        if arguments_delta:
            plan_content += f' args~ {arguments_delta}'
        self._write_status(f'[thinking] {plan_content}')

    def _render_tool_start(self, event: dict[str, object]) -> str:
        tool_name = event.get('tool_name')
        arguments = event.get('arguments')
        if not isinstance(tool_name, str) or not tool_name:
            return '[tool] starting'
        if not isinstance(arguments, dict):
            arguments = {}
        if tool_name == 'bash':
            command = _preview_value(arguments.get('command'))
            return f'[command] {command or "(empty command)"}'
        if tool_name in {'write_file', 'edit_file', 'read_file', 'notebook_edit'}:
            path = _preview_value(arguments.get('path'))
            return f'[file] {tool_name} {path or "(unknown path)"}'
        if tool_name == 'mcp_call_tool':
            server = _preview_value(arguments.get('server')) or '(auto)'
            remote_tool = _preview_value(arguments.get('tool_name')) or '(unknown)'
            tool_arguments = arguments.get('arguments')
            tool_args_preview = _preview_value(tool_arguments, max_chars=240)
            if tool_args_preview:
                return f'[mcp] server={server} tool={remote_tool} args={tool_args_preview}'
            return f'[mcp] server={server} tool={remote_tool}'
        if tool_name.startswith('mcp_'):
            summary = _preview_value(arguments)
            return f'[mcp] {tool_name} {summary}'.rstrip()
        summary = _preview_value(arguments)
        if summary:
            return f'[tool] {tool_name} {summary}'
        return f'[tool] {tool_name}'

    def _render_tool_delta(self, event: dict[str, object]) -> None:
        tool_name = event.get('tool_name')
        if not isinstance(tool_name, str) or not tool_name:
            return
        if tool_name != 'bash' and not tool_name.startswith('mcp_'):
            return
        delta = str(event.get('delta', ''))
        if not delta:
            return
        stream_name = event.get('stream')
        if not isinstance(stream_name, str) or not stream_name:
            stream_name = 'tool'
        key = (str(event.get('tool_call_id')), stream_name)
        self._close_assistant()
        if self._tool_stream_key != key:
            self._close_tool_stream()
            self._write(f'[tool-output:{tool_name}:{stream_name}]\n')
            self._tool_stream_key = key
            self._tool_stream_open = True
            self._tool_stream_ends_with_newline = True
        self._write(delta)
        self._tool_stream_open = True
        self._tool_stream_ends_with_newline = delta.endswith('\n')

    def _render_tool_result(self, event: dict[str, object]) -> str:
        tool_name = event.get('tool_name')
        ok = bool(event.get('ok'))
        metadata = event.get('metadata')
        content_preview = _preview_value(event.get('content_preview'), max_chars=220)
        if not isinstance(tool_name, str) or not tool_name:
            tool_name = 'tool'
        if not isinstance(metadata, dict):
            metadata = {}
        action = metadata.get('action')
        path = metadata.get('path')
        if action == 'bash':
            exit_code = metadata.get('exit_code')
            output_preview = _preview_value(metadata.get('output_preview'), max_chars=120)
            if output_preview:
                return f'[command] exit_code={exit_code} ok={ok} output={output_preview}'
            return f'[command] exit_code={exit_code} ok={ok}'
        if action == 'web_search':
            query = _preview_value(metadata.get('query'), max_chars=90)
            result_count = metadata.get('result_count')
            top_url = ''
            top_urls = metadata.get('top_urls')
            if isinstance(top_urls, list) and top_urls:
                top_url = _preview_value(top_urls[0], max_chars=120)
            parts = [f'[search] ok={ok}']
            if query:
                parts.append(f'query={query}')
            if isinstance(result_count, int):
                parts.append(f'results={result_count}')
            if top_url:
                parts.append(f'top={top_url}')
            return ' '.join(parts)
        if action == 'web_fetch':
            url = _preview_value(metadata.get('url'), max_chars=120)
            fetched_chars = metadata.get('fetched_chars')
            preview = _preview_value(metadata.get('preview'), max_chars=120)
            parts = [f'[web_fetch] ok={ok}']
            if url:
                parts.append(f'url={url}')
            if isinstance(fetched_chars, int):
                parts.append(f'chars={fetched_chars}')
            if metadata.get('truncated') is True:
                parts.append('truncated=True')
            if preview:
                parts.append(f'preview={preview}')
            return ' '.join(parts)
        if isinstance(path, str) and path:
            file_action = action if isinstance(action, str) and action else tool_name
            if file_action in {'write_file', 'edit_file', 'notebook_edit'}:
                label = 'updated'
            elif file_action == 'read_file':
                label = 'read'
            else:
                label = file_action
            return f'[file] {label} {path} ok={ok}'
        if action == 'mcp_call_tool' or tool_name.startswith('mcp_'):
            server = _preview_value(metadata.get('server_name') or metadata.get('requested_server')) or '(auto)'
            remote_tool = _preview_value(metadata.get('tool_name')) or tool_name
            summary = f'[mcp] server={server} tool={remote_tool} ok={ok}'
            if not ok and content_preview:
                summary += f' error={content_preview}'
            return summary
        cwd_update = metadata.get('cwd_update')
        if isinstance(cwd_update, str) and cwd_update:
            return f'[cwd] {cwd_update}'
        for candidate in (
            metadata.get('output_preview'),
            metadata.get('preview'),
            metadata.get('arguments_preview'),
            metadata.get('answer_preview'),
            content_preview,
        ):
            preview = _preview_value(candidate, max_chars=180)
            if preview:
                return f'[tool] {tool_name} ok={ok} {preview}'
        if isinstance(action, str) and action and action != tool_name:
            return f'[tool] {tool_name} action={action} ok={ok}'
        return f'[tool] {tool_name} ok={ok}'

    def _write_status(self, line: str) -> None:
        if not line:
            return
        self._close_open_blocks()
        self._write(line + '\n')

    def _close_open_blocks(self) -> None:
        self._close_assistant()
        self._close_tool_stream()

    def _close_assistant(self) -> None:
        if self._assistant_open and not self._assistant_ends_with_newline:
            self._write('\n')
        self._assistant_open = False
        self._assistant_ends_with_newline = True

    def _close_tool_stream(self) -> None:
        if self._tool_stream_open and not self._tool_stream_ends_with_newline:
            self._write('\n')
        self._tool_stream_key = None
        self._tool_stream_open = False
        self._tool_stream_ends_with_newline = True

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()


def _print_agent_result(
    result,
    *,
    show_transcript: bool,
    suppress_final_output: bool = False,
) -> None:
    if not suppress_final_output:
        print(result.final_output)
    print('\n# Usage')
    print(f'total_tokens={result.usage.total_tokens}')
    print(f'input_tokens={result.usage.input_tokens}')
    print(f'output_tokens={result.usage.output_tokens}')
    print(f'total_cost_usd={result.total_cost_usd:.6f}')
    if result.stop_reason:
        print(f'stop_reason={result.stop_reason}')
    if result.session_id:
        print('\n# Session')
        print(f'session_id={result.session_id}')
        if result.session_path:
            print(f'session_path={result.session_path}')
    if result.scratchpad_directory:
        print(f'scratchpad_directory={result.scratchpad_directory}')
    if show_transcript:
        print('\n# Transcript')
        for message in result.transcript:
            role = message.get('role', 'unknown')
            print(f'[{role}]')
            print(message.get('content', ''))


def _run_agent_turn(
    agent: LocalCodingAgent,
    prompt: str,
    *,
    show_transcript: bool,
    stored_session: StoredAgentSession | None = None,
    stream_output: TextIO | None = None,
    result_printer: Callable[..., None] = _print_agent_result,
):
    renderer = (
        _AgentLiveRenderer(stream_output)
        if agent.runtime_config.stream_model_responses
        else None
    )
    event_handler = renderer.handle_event if renderer is not None else None
    if stored_session is not None:
        result = agent.resume(
            prompt,
            stored_session,
            event_handler=event_handler,
        )
    else:
        result = agent.run(
            prompt,
            event_handler=event_handler,
        )
    if renderer is not None:
        renderer.finish()
    if result_printer is _print_agent_result:
        result_printer(
            result,
            show_transcript=show_transcript,
            suppress_final_output=bool(
                renderer is not None and renderer.streamed_assistant_output
            ),
        )
    else:
        result_printer(result, show_transcript=show_transcript)
    return result


def _run_agent_chat_loop(
    agent: LocalCodingAgent,
    *,
    initial_prompt: str | None,
    resume_session_id: str | None,
    show_transcript: bool,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    stream_output: TextIO | None = None,
    result_printer: Callable[..., None] = _print_agent_result,
) -> int:
    active_session_id = resume_session_id
    first_prompt = initial_prompt

    output_func('# Agent Chat')
    output_func("Enter a prompt. Use '/exit' or '/quit' to stop.")
    if active_session_id:
        output_func(f'resuming_session_id={active_session_id}')

    while True:
        if first_prompt is not None:
            prompt = first_prompt
            first_prompt = None
        else:
            try:
                prompt = input_func('user> ')
            except EOFError:
                output_func('chat_ended=eof')
                return 0
            except KeyboardInterrupt:
                output_func('\nchat_ended=interrupt')
                return 130

        normalized = prompt.strip()
        if not normalized:
            continue
        if normalized in {'/exit', '/quit'}:
            output_func('chat_ended=user_exit')
            return 0

        if active_session_id:
            stored_session = load_agent_session(
                active_session_id,
                directory=agent.runtime_config.session_directory,
            )
            result = _run_agent_turn(
                agent,
                prompt,
                show_transcript=show_transcript,
                stored_session=stored_session,
                stream_output=stream_output,
                result_printer=result_printer,
            )
        else:
            result = _run_agent_turn(
                agent,
                prompt,
                show_transcript=show_transcript,
                stream_output=stream_output,
                result_printer=result_printer,
            )
        active_session_id = result.session_id
