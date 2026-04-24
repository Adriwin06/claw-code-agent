from __future__ import annotations

import importlib
import sys

_COMPAT_MODULES = {
    'agent_context': 'src.agent.context.snapshot',
    'agent_context_usage': 'src.agent.context.usage',
    'agent_manager': 'src.agent.runtime.manager',
    'agent_plugin_cache': 'src.agent.plugins.cache',
    'agent_prompting': 'src.agent.context.prompting',
    'agent_registry': 'src.agent.profiles.registry',
    'agent_runtime': 'src.agent.runtime.agent',
    'agent_session': 'src.agent.models.session',
    'agent_slash_commands': 'src.agent.commands.slash',
    'agent_tools': 'src.agent.tools.execution',
    'agent_types': 'src.agent.models.types',
    'bash_security': 'src.agent.tools.bash_security',
    'builtin_agents': 'src.agent.profiles.builtin',
    'bundled_skills': 'src.agent.skills.bundled',
    'delegate_orchestrator': 'src.agent.runtime.delegation',
    'model_turn_runner': 'src.agent.runtime.model_turn',
    'prompt_constants': 'src.agent.context.constants',
    'prompt_pressure': 'src.agent.context.pressure',
    'run_state': 'src.agent.runtime.state',
    'runtime_dependencies': 'src.agent.runtime.dependencies',
    'session_persistence': 'src.agent.runtime.persistence',
    'tool_call_runner': 'src.agent.runtime.tool_calls',
}

for _name, _target in _COMPAT_MODULES.items():
    _module = importlib.import_module(_target)
    sys.modules[f'{__name__}.{_name}'] = _module
    globals()[_name] = _module

del importlib, sys, _name, _target, _module
