from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.agent.tools.execution import (
    AgentTool,
    _list_dir,
    _read_file,
    _write_file,
    _edit_file,
    _notebook_edit,
    _glob_search,
    _grep_search,
    _run_bash,
    _lsp_query,
    _web_fetch,
    _search_status,
    _search_list_providers,
    _search_activate_provider,
    _web_search,
    _tool_search,
    _list_available_tools,
    _sleep,
    _ask_user_question,
    _account_status,
    _account_list_profiles,
    _account_login,
    _account_logout,
    _config_list,
    _config_get,
    _config_set,
    _mcp_list_resources,
    _mcp_read_resource,
    _mcp_list_tools,
    _mcp_call_tool,
    _remote_status,
    _remote_list_profiles,
    _remote_connect,
    _remote_disconnect,
    _worktree_status,
    _worktree_enter,
    _worktree_exit,
    _workflow_list,
    _workflow_get,
    _workflow_run,
    _remote_trigger,
    _plan_get,
    _update_plan,
    _plan_clear,
    _task_next,
    _team_list,
    _team_get,
    _team_create,
    _team_delete,
    _send_message,
    _team_messages,
    _task_list,
    _task_get,
    _task_create,
    _task_update,
    _task_start,
    _task_complete,
    _task_block,
    _task_cancel,
    _enter_plan_mode,
    _exit_plan_mode,
    _task_output,
    _task_stop,
    _todo_write,
    _agent_tool_placeholder,
    _execute_skill,
)


def default_tool_registry() -> dict[str, AgentTool]:
    tools = [
        AgentTool(
            name='list_dir',
            description='List files and directories under a workspace path.',
            parameters={
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'Relative path from workspace root.'},
                    'max_entries': {'type': 'integer', 'minimum': 1, 'maximum': 500},
                },
            },
            handler=_list_dir,
        ),
        AgentTool(
            name='read_file',
            description='Read the contents of a UTF-8 text file inside the workspace.',
            parameters={
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'Relative file path from workspace root.'},
                    'start_line': {'type': 'integer', 'minimum': 1},
                    'end_line': {'type': 'integer', 'minimum': 1},
                },
                'required': ['path'],
            },
            handler=_read_file,
        ),
        AgentTool(
            name='write_file',
            description='Write a complete file inside the workspace. Creates parent directories when needed.',
            parameters={
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'content': {'type': 'string'},
                },
                'required': ['path', 'content'],
            },
            handler=_write_file,
        ),
        AgentTool(
            name='edit_file',
            description='Replace text inside a workspace file using exact string matching.',
            parameters={
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'old_text': {'type': 'string'},
                    'new_text': {'type': 'string'},
                    'replace_all': {'type': 'boolean'},
                },
                'required': ['path', 'old_text', 'new_text'],
            },
            handler=_edit_file,
        ),
        AgentTool(
            name='notebook_edit',
            description='Edit a Jupyter notebook cell by replacing or appending source in a .ipynb file.',
            parameters={
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'cell_index': {'type': 'integer', 'minimum': 0},
                    'source': {'type': 'string'},
                    'cell_type': {'type': 'string'},
                    'create_cell': {'type': 'boolean'},
                },
                'required': ['path', 'cell_index', 'source'],
            },
            handler=_notebook_edit,
        ),
        AgentTool(
            name='glob_search',
            description='Find files matching a glob pattern inside the workspace.',
            parameters={
                'type': 'object',
                'properties': {
                    'pattern': {'type': 'string'},
                },
                'required': ['pattern'],
            },
            handler=_glob_search,
        ),
        AgentTool(
            name='grep_search',
            description='Search for a string or regular expression inside workspace files.',
            parameters={
                'type': 'object',
                'properties': {
                    'pattern': {'type': 'string'},
                    'path': {'type': 'string'},
                    'literal': {'type': 'boolean'},
                    'max_matches': {'type': 'integer', 'minimum': 1, 'maximum': 500},
                },
                'required': ['pattern'],
            },
            handler=_grep_search,
        ),
        AgentTool(
            name='bash',
            description='Run a shell command in the workspace. Use sparingly and prefer dedicated file tools for edits.',
            parameters={
                'type': 'object',
                'properties': {
                    'command': {'type': 'string'},
                },
                'required': ['command'],
            },
            handler=_run_bash,
        ),
        AgentTool(
            name='LSP',
            description='Use local LSP-style code intelligence for definitions, references, hover, symbols, and call hierarchy.',
            parameters={
                'type': 'object',
                'properties': {
                    'operation': {
                        'type': 'string',
                        'enum': [
                            'goToDefinition',
                            'findReferences',
                            'hover',
                            'documentSymbol',
                            'workspaceSymbol',
                            'goToImplementation',
                            'prepareCallHierarchy',
                            'incomingCalls',
                            'outgoingCalls',
                        ],
                    },
                    'file_path': {'type': 'string'},
                    'line': {'type': 'integer', 'minimum': 1},
                    'character': {'type': 'integer', 'minimum': 1},
                    'query': {'type': 'string'},
                    'max_results': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
                'required': ['operation', 'file_path', 'line', 'character'],
            },
            handler=_lsp_query,
        ),
        AgentTool(
            name='web_fetch',
            description='Fetch a text resource from http, https, or file URLs and return a truncated text response.',
            parameters={
                'type': 'object',
                'properties': {
                    'url': {'type': 'string'},
                    'max_chars': {'type': 'integer', 'minimum': 1, 'maximum': 100000},
                },
                'required': ['url'],
            },
            handler=_web_fetch,
        ),
        AgentTool(
            name='search_status',
            description='Show the local search runtime summary or a specific configured search provider.',
            parameters={
                'type': 'object',
                'properties': {
                    'provider': {'type': 'string'},
                },
            },
            handler=_search_status,
        ),
        AgentTool(
            name='search_list_providers',
            description='List configured local search providers from workspace search manifests and environment configuration.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'max_providers': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_search_list_providers,
        ),
        AgentTool(
            name='search_activate_provider',
            description='Set the active local search provider for the current workspace.',
            parameters={
                'type': 'object',
                'properties': {
                    'provider': {'type': 'string'},
                },
                'required': ['provider'],
            },
            handler=_search_activate_provider,
        ),
        AgentTool(
            name='web_search',
            description='Run a real web search against a configured search backend and return ranked results.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'provider': {'type': 'string'},
                    'domains': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                    'max_results': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                },
                'required': ['query'],
            },
            handler=_web_search,
        ),
        AgentTool(
            name='tool_search',
            description='Search the active tool registry by tool name or description.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'max_results': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                },
                'required': ['query'],
            },
            handler=_tool_search,
        ),
        AgentTool(
            name='list_available_tools',
            description='List the tool names currently available in this session.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'max_results': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                },
            },
            handler=_list_available_tools,
        ),
        AgentTool(
            name='sleep',
            description='Pause execution briefly for bounded local wait flows.',
            parameters={
                'type': 'object',
                'properties': {
                    'seconds': {'type': 'number', 'minimum': 0.0, 'maximum': 5.0},
                },
                'required': ['seconds'],
            },
            handler=_sleep,
        ),
        AgentTool(
            name='ask_user_question',
            description='Request an answer from the local ask-user runtime using queued or interactive answers.',
            parameters={
                'type': 'object',
                'properties': {
                    'question': {'type': 'string'},
                    'header': {'type': 'string'},
                    'question_id': {'type': 'string'},
                    'choices': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                    'allow_free_text': {'type': 'boolean'},
                },
                'required': ['question'],
            },
            handler=_ask_user_question,
        ),
        AgentTool(
            name='account_status',
            description='Show local account runtime summary or a specific configured account profile.',
            parameters={
                'type': 'object',
                'properties': {
                    'profile': {'type': 'string'},
                },
            },
            handler=_account_status,
        ),
        AgentTool(
            name='account_list_profiles',
            description='List configured local account profiles from workspace account manifests.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'max_profiles': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_account_list_profiles,
        ),
        AgentTool(
            name='account_login',
            description='Activate a local account profile or ephemeral account identity and persist it as the active account session.',
            parameters={
                'type': 'object',
                'properties': {
                    'target': {'type': 'string'},
                    'provider': {'type': 'string'},
                    'auth_mode': {'type': 'string'},
                },
                'required': ['target'],
            },
            handler=_account_login,
        ),
        AgentTool(
            name='account_logout',
            description='Clear the active local account session state.',
            parameters={
                'type': 'object',
                'properties': {
                    'reason': {'type': 'string'},
                },
            },
            handler=_account_logout,
        ),
        AgentTool(
            name='config_list',
            description='List merged or source-specific workspace config keys from local settings files.',
            parameters={
                'type': 'object',
                'properties': {
                    'source': {'type': 'string'},
                    'prefix': {'type': 'string'},
                    'limit': {'type': 'integer', 'minimum': 1, 'maximum': 500},
                },
            },
            handler=_config_list,
        ),
        AgentTool(
            name='config_get',
            description='Read a merged or source-specific workspace config value by dotted key path.',
            parameters={
                'type': 'object',
                'properties': {
                    'key_path': {'type': 'string'},
                    'source': {'type': 'string'},
                },
                'required': ['key_path'],
            },
            handler=_config_get,
        ),
        AgentTool(
            name='config_set',
            description='Write a workspace config value by dotted key path into a chosen config source.',
            parameters={
                'type': 'object',
                'properties': {
                    'key_path': {'type': 'string'},
                    'source': {'type': 'string'},
                    'value': {
                        'oneOf': [
                            {'type': 'string'},
                            {'type': 'number'},
                            {'type': 'integer'},
                            {'type': 'boolean'},
                            {'type': 'array'},
                            {'type': 'object'},
                            {'type': 'null'},
                        ]
                    },
                },
                'required': ['key_path', 'value'],
            },
            handler=_config_set,
        ),
        AgentTool(
            name='mcp_list_resources',
            description='List local MCP resources discovered from workspace MCP manifests.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'max_resources': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_mcp_list_resources,
        ),
        AgentTool(
            name='mcp_read_resource',
            description='Read a local MCP resource by URI from workspace MCP manifests.',
            parameters={
                'type': 'object',
                'properties': {
                    'uri': {'type': 'string'},
                    'max_chars': {'type': 'integer', 'minimum': 1, 'maximum': 50000},
                },
                'required': ['uri'],
            },
            handler=_mcp_read_resource,
        ),
        AgentTool(
            name='mcp_list_tools',
            description='List MCP tools exposed by configured MCP servers over real transport.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'server': {'type': 'string'},
                    'max_tools': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_mcp_list_tools,
        ),
        AgentTool(
            name='mcp_call_tool',
            description='Call an MCP tool exposed by a configured MCP server over real transport.',
            parameters={
                'type': 'object',
                'properties': {
                    'tool_name': {'type': 'string'},
                    'server': {'type': 'string'},
                    'arguments': {'type': 'object'},
                    'max_chars': {'type': 'integer', 'minimum': 1, 'maximum': 50000},
                },
                'required': ['tool_name'],
            },
            handler=_mcp_call_tool,
        ),
        AgentTool(
            name='remote_status',
            description='Show the local remote runtime summary or a specific configured remote profile.',
            parameters={
                'type': 'object',
                'properties': {
                    'profile': {'type': 'string'},
                },
            },
            handler=_remote_status,
        ),
        AgentTool(
            name='remote_list_profiles',
            description='List configured local remote profiles from workspace remote manifests.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'mode': {'type': 'string'},
                    'max_profiles': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_remote_list_profiles,
        ),
        AgentTool(
            name='remote_connect',
            description='Activate a local remote target or configured remote profile and persist it as the active connection.',
            parameters={
                'type': 'object',
                'properties': {
                    'target': {'type': 'string'},
                    'mode': {'type': 'string'},
                },
                'required': ['target'],
            },
            handler=_remote_connect,
        ),
        AgentTool(
            name='remote_disconnect',
            description='Clear the active local remote connection state.',
            parameters={
                'type': 'object',
                'properties': {
                    'reason': {'type': 'string'},
                },
            },
            handler=_remote_disconnect,
        ),
        AgentTool(
            name='worktree_status',
            description='Show the current managed git worktree session status.',
            parameters={
                'type': 'object',
                'properties': {},
            },
            handler=_worktree_status,
        ),
        AgentTool(
            name='worktree_enter',
            description='Create an isolated git worktree and switch the current agent session into it.',
            parameters={
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                },
            },
            handler=_worktree_enter,
        ),
        AgentTool(
            name='worktree_exit',
            description='Leave the active managed worktree session and optionally remove the worktree.',
            parameters={
                'type': 'object',
                'properties': {
                    'action': {'type': 'string'},
                    'discard_changes': {'type': 'boolean'},
                },
            },
            handler=_worktree_exit,
        ),
        AgentTool(
            name='workflow_list',
            description='List local workflow definitions discovered from workspace workflow manifests.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'max_workflows': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_workflow_list,
        ),
        AgentTool(
            name='workflow_get',
            description='Show one local workflow definition by name.',
            parameters={
                'type': 'object',
                'properties': {
                    'workflow_name': {'type': 'string'},
                },
                'required': ['workflow_name'],
            },
            handler=_workflow_get,
        ),
        AgentTool(
            name='workflow_run',
            description='Record and render a local workflow execution request from a workflow manifest.',
            parameters={
                'type': 'object',
                'properties': {
                    'workflow_name': {'type': 'string'},
                    'arguments': {'type': 'object'},
                },
                'required': ['workflow_name'],
            },
            handler=_workflow_run,
        ),
        AgentTool(
            name='remote_trigger',
            description='List, inspect, create, update, or run local remote triggers similar to the npm remote trigger tool.',
            parameters={
                'type': 'object',
                'properties': {
                    'action': {'type': 'string'},
                    'trigger_id': {'type': 'string'},
                    'body': {'type': 'object'},
                    'query': {'type': 'string'},
                    'max_triggers': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
                'required': ['action'],
            },
            handler=_remote_trigger,
        ),
        AgentTool(
            name='plan_get',
            description='Show the current local runtime plan.',
            parameters={
                'type': 'object',
                'properties': {},
            },
            handler=_plan_get,
        ),
        AgentTool(
            name='update_plan',
            description='Replace the current local runtime plan with a structured multi-step plan and optionally sync it to tasks.',
            parameters={
                'type': 'object',
                'properties': {
                    'explanation': {'type': 'string'},
                    'sync_tasks': {'type': 'boolean'},
                    'items': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'step': {'type': 'string'},
                                'status': {'type': 'string'},
                                'task_id': {'type': 'string'},
                                'description': {'type': 'string'},
                                'priority': {'type': 'string'},
                                'active_form': {'type': 'string'},
                                'owner': {'type': 'string'},
                                'depends_on': {
                                    'type': 'array',
                                    'items': {'type': 'string'},
                                },
                            },
                            'required': ['step'],
                        },
                    },
                },
                'required': ['items'],
            },
            handler=_update_plan,
        ),
        AgentTool(
            name='plan_clear',
            description='Clear the current local runtime plan and optionally sync the task runtime.',
            parameters={
                'type': 'object',
                'properties': {
                    'sync_tasks': {'type': 'boolean'},
                },
            },
            handler=_plan_clear,
        ),
        AgentTool(
            name='task_next',
            description='Show the next actionable tasks from the local runtime task list.',
            parameters={
                'type': 'object',
                'properties': {
                    'max_tasks': {'type': 'integer', 'minimum': 1, 'maximum': 50},
                },
            },
            handler=_task_next,
        ),
        AgentTool(
            name='team_list',
            description='List locally configured collaboration teams.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'max_teams': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_team_list,
        ),
        AgentTool(
            name='team_get',
            description='Show a locally configured collaboration team by name.',
            parameters={
                'type': 'object',
                'properties': {
                    'team_name': {'type': 'string'},
                },
                'required': ['team_name'],
            },
            handler=_team_get,
        ),
        AgentTool(
            name='team_create',
            description='Create a locally stored collaboration team.',
            parameters={
                'type': 'object',
                'properties': {
                    'team_name': {'type': 'string'},
                    'description': {'type': 'string'},
                    'members': {'type': 'array', 'items': {'type': 'string'}},
                    'metadata': {'type': 'object'},
                },
                'required': ['team_name'],
            },
            handler=_team_create,
        ),
        AgentTool(
            name='team_delete',
            description='Delete a locally stored collaboration team and its recorded messages.',
            parameters={
                'type': 'object',
                'properties': {
                    'team_name': {'type': 'string'},
                },
                'required': ['team_name'],
            },
            handler=_team_delete,
        ),
        AgentTool(
            name='send_message',
            description='Send a local collaboration message to a team or teammate and persist it in the team runtime.',
            parameters={
                'type': 'object',
                'properties': {
                    'team_name': {'type': 'string'},
                    'message': {'type': 'string'},
                    'sender': {'type': 'string'},
                    'recipient': {'type': 'string'},
                    'metadata': {'type': 'object'},
                },
                'required': ['team_name', 'message'],
            },
            handler=_send_message,
        ),
        AgentTool(
            name='team_messages',
            description='Show locally recorded collaboration messages for all teams or one team.',
            parameters={
                'type': 'object',
                'properties': {
                    'team_name': {'type': 'string'},
                    'limit': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_team_messages,
        ),
        AgentTool(
            name='task_list',
            description='List locally stored runtime tasks.',
            parameters={
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'owner': {'type': 'string'},
                    'actionable_only': {'type': 'boolean'},
                    'max_tasks': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_task_list,
        ),
        AgentTool(
            name='task_get',
            description='Show a locally stored runtime task by id.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                },
                'required': ['task_id'],
            },
            handler=_task_get,
        ),
        AgentTool(
            name='task_create',
            description='Create a locally stored runtime task.',
            parameters={
                'type': 'object',
                'properties': {
                    'title': {'type': 'string'},
                    'description': {'type': 'string'},
                    'status': {'type': 'string'},
                    'priority': {'type': 'string'},
                    'task_id': {'type': 'string'},
                    'active_form': {'type': 'string'},
                    'owner': {'type': 'string'},
                    'blocks': {'type': 'array', 'items': {'type': 'string'}},
                    'blocked_by': {'type': 'array', 'items': {'type': 'string'}},
                    'metadata': {'type': 'object'},
                },
                'required': ['title'],
            },
            handler=_task_create,
        ),
        AgentTool(
            name='task_update',
            description='Update a locally stored runtime task by id.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                    'title': {'type': 'string'},
                    'description': {'type': 'string'},
                    'status': {'type': 'string'},
                    'priority': {'type': 'string'},
                    'active_form': {'type': 'string'},
                    'owner': {'type': 'string'},
                    'blocks': {'type': 'array', 'items': {'type': 'string'}},
                    'blocked_by': {'type': 'array', 'items': {'type': 'string'}},
                    'metadata': {'type': 'object'},
                },
                'required': ['task_id'],
            },
            handler=_task_update,
        ),
        AgentTool(
            name='task_start',
            description='Mark a task as in progress, or blocked if dependencies are unresolved.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                    'owner': {'type': 'string'},
                    'active_form': {'type': 'string'},
                },
                'required': ['task_id'],
            },
            handler=_task_start,
        ),
        AgentTool(
            name='task_complete',
            description='Mark a task as completed and release blocked dependents when possible.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                },
                'required': ['task_id'],
            },
            handler=_task_complete,
        ),
        AgentTool(
            name='task_block',
            description='Mark a task as blocked with optional dependencies and reason.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                    'blocked_by': {'type': 'array', 'items': {'type': 'string'}},
                    'reason': {'type': 'string'},
                },
                'required': ['task_id'],
            },
            handler=_task_block,
        ),
        AgentTool(
            name='task_cancel',
            description='Mark a task as cancelled with an optional reason.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                    'reason': {'type': 'string'},
                },
                'required': ['task_id'],
            },
            handler=_task_cancel,
        ),
        AgentTool(
            name='EnterPlanMode',
            description=(
                'Enter plan mode. In plan mode, focus on exploring the codebase '
                'and creating a plan rather than making changes. Use read-only '
                'tools (Read, Grep, Glob) to investigate, then create a plan.'
            ),
            parameters={
                'type': 'object',
                'properties': {},
            },
            handler=_enter_plan_mode,
        ),
        AgentTool(
            name='ExitPlanMode',
            description=(
                'Exit plan mode and return to normal execution mode. '
                'Call this after you have finished exploring and have a plan ready.'
            ),
            parameters={
                'type': 'object',
                'properties': {},
            },
            handler=_exit_plan_mode,
        ),
        AgentTool(
            name='TaskOutput',
            description=(
                'Get the output of a background task by its ID. '
                'Use block=true (default) to wait for completion, '
                'or block=false to check current status without waiting.'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {
                        'type': 'string',
                        'description': 'The task ID to get output from.',
                    },
                    'block': {
                        'type': 'boolean',
                        'description': 'Whether to wait for completion (default true).',
                    },
                    'timeout': {
                        'type': 'number',
                        'description': 'Max wait time in ms (0-600000, default 30000).',
                    },
                },
                'required': ['task_id'],
            },
            handler=_task_output,
        ),
        AgentTool(
            name='TaskStop',
            description='Stop a running background task by its ID.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {
                        'type': 'string',
                        'description': 'The task ID to stop.',
                    },
                },
                'required': ['task_id'],
            },
            handler=_task_stop,
        ),
        AgentTool(
            name='todo_write',
            description='Replace the current local runtime task list with a structured todo list.',
            parameters={
                'type': 'object',
                'properties': {
                    'items': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'task_id': {'type': 'string'},
                                'title': {'type': 'string'},
                                'description': {'type': 'string'},
                                'status': {'type': 'string'},
                                'priority': {'type': 'string'},
                                'active_form': {'type': 'string'},
                                'owner': {'type': 'string'},
                                'blocks': {'type': 'array', 'items': {'type': 'string'}},
                                'blocked_by': {'type': 'array', 'items': {'type': 'string'}},
                                'metadata': {'type': 'object'},
                            },
                            'required': ['title'],
                        },
                    },
                },
                'required': ['items'],
            },
            handler=_todo_write,
        ),
        AgentTool(
            name='Agent',
            description=(
                'Launch a new agent to handle complex, multi-step tasks. '
                'Each agent type has specific capabilities and tools available to it.'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'description': {
                        'type': 'string',
                        'description': 'A short (3-5 word) description of the task',
                    },
                    'prompt': {
                        'type': 'string',
                        'description': 'The task for the agent to perform',
                    },
                    'subagent_type': {
                        'type': 'string',
                        'description': 'The type of specialized agent to use for this task',
                    },
                    'model': {
                        'type': 'string',
                        'enum': ['sonnet', 'opus', 'haiku'],
                        'description': 'Optional model override for this agent',
                    },
                    'run_in_background': {
                        'type': 'boolean',
                        'description': 'Set to true to run this agent in the background',
                    },
                    'isolation': {
                        'type': 'string',
                        'enum': ['worktree'],
                        'description': 'Isolation mode. "worktree" creates a temporary git worktree.',
                    },
                    'subtasks': {
                        'type': 'array',
                        'description': (
                            'Multiple child-agent subtasks. Use this instead of prompt when '
                            'several independent agents should be launched from one tool call.'
                        ),
                        'items': {
                            'oneOf': [
                                {'type': 'string'},
                                {
                                    'type': 'object',
                                    'properties': {
                                        'prompt': {'type': 'string'},
                                        'label': {'type': 'string'},
                                        'max_turns': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                                        'resume_session_id': {'type': 'string'},
                                        'session_id': {'type': 'string'},
                                        'depends_on': {
                                            'type': 'array',
                                            'items': {'type': 'string'},
                                        },
                                    },
                                    'required': ['prompt'],
                                },
                            ]
                        },
                    },
                    'resume_session_id': {'type': 'string'},
                    'session_id': {'type': 'string'},
                    'max_turns': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                    'allow_write': {'type': 'boolean'},
                    'allow_shell': {'type': 'boolean'},
                    'include_parent_context': {'type': 'boolean'},
                    'continue_on_error': {'type': 'boolean'},
                    'max_failures': {'type': 'integer', 'minimum': 0, 'maximum': 20},
                    'strategy': {
                        'type': 'string',
                        'enum': ['serial', 'parallel', 'topological'],
                        'description': (
                            'serial runs subtasks one after another; parallel runs independent '
                            'subtasks concurrently; topological runs dependency batches in order.'
                        ),
                    },
                    'max_parallel_subtasks': {
                        'type': 'integer',
                        'minimum': 1,
                        'maximum': 8,
                        'description': 'Maximum concurrent child agents for parallel batches.',
                    },
                },
                'required': ['description'],
            },
            handler=_agent_tool_placeholder,
        ),
        # Keep legacy name for backward compatibility
        AgentTool(
            name='delegate_agent',
            description='(Legacy) Delegate a subtask to a nested agent. Prefer using the Agent tool instead.',
            parameters={
                'type': 'object',
                'properties': {
                    'prompt': {'type': 'string'},
                    'subtasks': {
                        'type': 'array',
                        'description': (
                            'Multiple child-agent subtasks. Use this instead of prompt when '
                            'several independent agents should be launched from one tool call.'
                        ),
                        'items': {
                            'oneOf': [
                                {'type': 'string'},
                                {
                                    'type': 'object',
                                    'properties': {
                                        'prompt': {'type': 'string'},
                                        'label': {'type': 'string'},
                                        'max_turns': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                                        'resume_session_id': {'type': 'string'},
                                        'session_id': {'type': 'string'},
                                        'depends_on': {
                                            'type': 'array',
                                            'items': {'type': 'string'},
                                        },
                                    },
                                    'required': ['prompt'],
                                },
                            ]
                        },
                    },
                    'resume_session_id': {'type': 'string'},
                    'session_id': {'type': 'string'},
                    'max_turns': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                    'allow_write': {'type': 'boolean'},
                    'allow_shell': {'type': 'boolean'},
                    'include_parent_context': {'type': 'boolean'},
                    'continue_on_error': {'type': 'boolean'},
                    'max_failures': {'type': 'integer', 'minimum': 0, 'maximum': 20},
                    'strategy': {
                        'type': 'string',
                        'enum': ['serial', 'parallel', 'topological'],
                        'description': (
                            'serial runs subtasks one after another; parallel runs independent '
                            'subtasks concurrently; topological runs dependency batches in order.'
                        ),
                    },
                    'max_parallel_subtasks': {
                        'type': 'integer',
                        'minimum': 1,
                        'maximum': 8,
                        'description': 'Maximum concurrent child agents for parallel batches.',
                    },
                },
            },
            handler=_agent_tool_placeholder,
        ),
        AgentTool(
            name='Skill',
            description=(
                'Execute a skill within the main conversation. '
                'Skills provide specialized capabilities and domain knowledge.'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'skill': {
                        'type': 'string',
                        'description': 'The skill name. E.g., "commit", "compact", or "help"',
                    },
                    'args': {
                        'type': 'string',
                        'description': 'Optional arguments for the skill',
                    },
                },
                'required': ['skill'],
            },
            handler=_execute_skill,
        ),
    ]
    return {tool.name: tool for tool in tools}



def build_effective_tool_registry(
    base_registry: Mapping[str, AgentTool] | None = None,
    *,
    plugin_runtime: Any = None,
) -> dict[str, AgentTool]:
    registry = dict(base_registry or default_tool_registry())
    if plugin_runtime is None:
        return registry

    alias_tools = plugin_runtime.register_tool_aliases(registry)
    if alias_tools:
        registry = {**registry, **alias_tools}
    virtual_tools = plugin_runtime.register_virtual_tools(registry)
    if virtual_tools:
        registry = {**registry, **virtual_tools}
    return registry
