# Claw Code Agent Docs

Open `index.html` in a browser to inspect the static architecture and runtime guide.

The site is intentionally build-free:

- `index.html` contains the document shell, project overview, language toggle, runbook, and section anchors.
- `styles.css` owns the responsive layout, light/dark themes, graph styling, flow cards, and event groups.
- `app.js` contains the source-linked data model and renders project facts, architecture, permission model, state/resume map, control surface map, commands, tools, runtime timeline, context graph, budget flow, event groups, and module atlas.

The data is grounded in the current Python runtime under `src/`, plus `tests/`,
`benchmarks/`, top-level project metadata, and the CLI/tool/slash registries.
It avoids subjective percentages and volatile per-file line counts.
