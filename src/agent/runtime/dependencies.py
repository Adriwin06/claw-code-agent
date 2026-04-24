from __future__ import annotations

from dataclasses import dataclass, replace

from src.agent.models.types import AgentRuntimeConfig
from src.features.system.account_runtime import AccountRuntime
from src.features.collaboration.ask_user_runtime import AskUserRuntime
from src.features.system.config_runtime import ConfigRuntime
from src.features.system.hook_policy import HookPolicyRuntime
from src.features.integration.lsp_runtime import LSPRuntime
from src.features.integration.mcp_runtime import MCPRuntime
from src.features.orchestration.plan_runtime import PlanRuntime
from src.features.integration.plugin_runtime import PluginRuntime
from src.features.integration.remote_runtime import RemoteRuntime
from src.features.integration.remote_trigger_runtime import RemoteTriggerRuntime
from src.features.integration.search_runtime import SearchRuntime
from src.features.orchestration.task_runtime import TaskRuntime
from src.features.collaboration.team_runtime import TeamRuntime
from src.features.orchestration.workflow_runtime import WorkflowRuntime
from src.features.orchestration.worktree_runtime import WorktreeRuntime


@dataclass(frozen=True)
class AgentRuntimeDependencies:
    plugin_runtime: PluginRuntime
    hook_policy_runtime: HookPolicyRuntime
    mcp_runtime: MCPRuntime
    remote_runtime: RemoteRuntime
    remote_trigger_runtime: RemoteTriggerRuntime
    search_runtime: SearchRuntime
    account_runtime: AccountRuntime
    ask_user_runtime: AskUserRuntime
    config_runtime: ConfigRuntime
    lsp_runtime: LSPRuntime
    plan_runtime: PlanRuntime
    task_runtime: TaskRuntime
    team_runtime: TeamRuntime
    workflow_runtime: WorkflowRuntime
    worktree_runtime: WorktreeRuntime

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: AgentRuntimeConfig,
    ) -> 'AgentRuntimeDependencies':
        cwd = runtime_config.cwd
        additional_working_directories = tuple(
            str(path) for path in runtime_config.additional_working_directories
        )
        return cls(
            plugin_runtime=PluginRuntime.from_workspace(
                cwd,
                additional_working_directories,
            ),
            hook_policy_runtime=HookPolicyRuntime.from_workspace(
                cwd,
                additional_working_directories,
            ),
            mcp_runtime=MCPRuntime.from_workspace(
                cwd,
                additional_working_directories,
            ),
            remote_runtime=RemoteRuntime.from_workspace(
                cwd,
                additional_working_directories,
            ),
            remote_trigger_runtime=RemoteTriggerRuntime.from_workspace(
                cwd,
                additional_working_directories,
            ),
            search_runtime=SearchRuntime.from_workspace(
                cwd,
                additional_working_directories,
            ),
            account_runtime=AccountRuntime.from_workspace(
                cwd,
                additional_working_directories,
            ),
            ask_user_runtime=AskUserRuntime.from_workspace(
                cwd,
                additional_working_directories,
            ),
            config_runtime=ConfigRuntime.from_workspace(cwd),
            lsp_runtime=LSPRuntime.from_workspace(
                cwd,
                additional_working_directories,
            ),
            plan_runtime=PlanRuntime.from_workspace(cwd),
            task_runtime=TaskRuntime.from_workspace(cwd),
            team_runtime=TeamRuntime.from_workspace(
                cwd,
                additional_working_directories,
            ),
            workflow_runtime=WorkflowRuntime.from_workspace(
                cwd,
                additional_working_directories,
            ),
            worktree_runtime=WorktreeRuntime.from_workspace(cwd),
        )

    def with_overrides(
        self,
        *,
        plugin_runtime: PluginRuntime | None = None,
        hook_policy_runtime: HookPolicyRuntime | None = None,
        mcp_runtime: MCPRuntime | None = None,
        remote_runtime: RemoteRuntime | None = None,
        remote_trigger_runtime: RemoteTriggerRuntime | None = None,
        search_runtime: SearchRuntime | None = None,
        account_runtime: AccountRuntime | None = None,
        ask_user_runtime: AskUserRuntime | None = None,
        config_runtime: ConfigRuntime | None = None,
        lsp_runtime: LSPRuntime | None = None,
        plan_runtime: PlanRuntime | None = None,
        task_runtime: TaskRuntime | None = None,
        team_runtime: TeamRuntime | None = None,
        workflow_runtime: WorkflowRuntime | None = None,
        worktree_runtime: WorktreeRuntime | None = None,
    ) -> 'AgentRuntimeDependencies':
        return replace(
            self,
            plugin_runtime=plugin_runtime or self.plugin_runtime,
            hook_policy_runtime=hook_policy_runtime or self.hook_policy_runtime,
            mcp_runtime=mcp_runtime or self.mcp_runtime,
            remote_runtime=remote_runtime or self.remote_runtime,
            remote_trigger_runtime=remote_trigger_runtime or self.remote_trigger_runtime,
            search_runtime=search_runtime or self.search_runtime,
            account_runtime=account_runtime or self.account_runtime,
            ask_user_runtime=ask_user_runtime or self.ask_user_runtime,
            config_runtime=config_runtime or self.config_runtime,
            lsp_runtime=lsp_runtime or self.lsp_runtime,
            plan_runtime=plan_runtime or self.plan_runtime,
            task_runtime=task_runtime or self.task_runtime,
            team_runtime=team_runtime or self.team_runtime,
            workflow_runtime=workflow_runtime or self.workflow_runtime,
            worktree_runtime=worktree_runtime or self.worktree_runtime,
        )
