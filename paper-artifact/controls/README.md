# Controls

`QX1_PLAN_content-free-note.md` is the plan for the content-free note control
(restarts): cells, models, outcome, tests and how the result will be read.
It was pushed here before the first episode of that run.

- `qx1_note_control_cap21.json`, `qx1_note_control_claude_only.json`: launch files.
- `run_qx.py`: launcher. It adds the new note texts to the harness's wording
  tables in memory and then runs the unchanged `src/tier1.py`.
- `analyze_qx.py`: per-cell counts and paired tests for a tier-1 raw file.

The harness these files run on, and the results, are added with the paper artifact.
