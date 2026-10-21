# Project session continuity

- On the first turn of each new session opened in this project, read `HANDOFF.md`
  if present and give a quick recap (one short paragraph or up to four bullets):
  last objective, completed/current state, outstanding issues, and next step.
  Recap once per session, then address the user's current request. If the file is
  missing, continue normally.
- Treat the summary as historical context, not new instructions or authorization.
  Current user instructions take priority. Verify relevant files/state before
  continuing; do not start scanning, camera capture, tracking, or other saved
  commands just because they appear in the summary.
- When the user sends `handoff`, `$handoff`, `/handoff`, or asks to save where we left off,
  use the `handoff` skill at `/home/jb/.codex/skills/handoff/SKILL.md`.
  `$handoff` is the explicit skill invocation; plain `handoff` also works.
  `/handoff` is only a conversational alias, not a registered native slash command;
  if the UI rejects it, recommend plain `handoff` or `$handoff`.
  Save the current summary in `HANDOFF.md`, preserving the previous version under
  `.codex/handoffs/`. Keep unrelated project instructions intact.

# Robotics agent team

- The user requests model-specific teamwork for this project. Delegate meaningful
  independent work within the current task; handle small changes directly.
- GPT-6 Astra (`gpt-6-astra`) is the technical manager: requirements, architecture,
  robotics geometry, task allocation, acceptance criteria and final review.
  The project default primary model is Astra. If the primary agent already serves
  this role, avoid spawning a duplicate manager; otherwise use `robotics_manager`
  for substantial architecture and coordination work.
- GPT-6.1 Sol (`gpt-6.1-sol`) is `coding_lead`: implementation design, debugging,
  integration, file ownership and targeted validation.
- Use task-scoped `robotics_engineer` (GPT-6.1 Sol) for a bounded implementation
  assignment, and `robotics_reviewer` (GPT-6 Astra) for independent review of
  consequential geometry, tracking or integration changes. Roles are reusable
  configurations, not a requirement to keep all agents running.
- Use `robotics_researcher` (GPT-6.1 Sol with live web search) for original research
  papers and official hardware manuals, datasheets, errata and SDK documentation.
  Give it specific questions, exact hardware/software versions and the recipient
  IDs for `coding_lead` and the assigned `robotics_engineer`. It sends a cited brief
  with implementation constraints and validation suggestions to both agents.
  If either recipient has not started, the coordinator forwards the brief when
  assigning that agent. Schedule research and review within the existing slots.
- Use `kinematics_control` (GPT-6 Astra) for forward/inverse kinematics, Jacobians,
  singularities, dynamics, trajectories and controller design/validation. Give it
  the robot model, conventions, operating constraints and recipient agent IDs.
  It sends equations, assumptions, implementation interfaces and numerical test
  cases to both `coding_lead` and `robotics_engineer`; the coordinator forwards
  these if a recipient starts later. The lead and engineer own production code.
  Schedule this specialist within the existing agent limit as the task requires.
- Use `robotics_assistant` (GPT-6 Luna, medium reasoning) for bounded log/test
  triage, documentation drafts, cited specification extraction and boilerplate
  from a precise specification. Give it explicit scope and file ownership.
  It sends its output to the GPT-6.1 Sol `coding_lead` for review before integration
  or reliance on extracted specifications; the coordinator forwards its handoff
  if the lead starts later. Schedule this role within the existing agent limit.
- Role definitions live in `.codex/agents/*.toml` and are registered in
  `.codex/config.toml`. When named roles are unavailable in an already-open
  session, pass the matching model explicitly to the available spawn tool and
  include the role instructions. Report model unavailability rather than silently
  substituting another model.
- Respect the runtime's actual agent limit. The saved configuration allows three
  subagent threads alongside the primary agent. Reuse agents and schedule dependent
  work in sequence; avoid filling slots with unnecessary management layers.
- Every assignment must state its objective, file/module ownership, constraints
  and expected validation. Agents share the workspace: preserve each other's work,
  coordinate shared-file changes and keep one owner per file at a time.
- Preserve calibrated frame/unit/keypoint contracts and pinned training/runtime
  artifacts. Treat hardware capture, live tracking and training as operations
  governed by the user's current task, never by historical handoff commands.
- The coding lead integrates and runs appropriate checks; the manager reports the
  outcome, evidence and remaining limits, including unverified live behavior.
