# Controls

`QX1_PLAN_content-free-note.md` is the plan for the content-free note control
(restarts): cells, models, outcome, tests and how the result will be read.
It was pushed here before the first episode of that run.

- `qx1_note_control_cap21.json`, `qx1_note_control_claude_only.json`: launch files.
- `run_qx.py`: launcher. It adds the new note texts to the harness's wording
  tables in memory and then runs the unchanged `src/tier1.py`.
- `analyze_qx.py`: per-cell counts and paired tests for a tier-1 raw file.
- `qx1_analysis.py`: the analysis the plan fixes (cells, the two one-sided tests
  with Holm correction, the descriptive contrasts). Pushed while the run was in
  progress, before any of its results were read.

## What ran (11 November 2026)

- Plan pushed in commit `8c1a9c1` at 02:07 UTC. First episode at 02:11 UTC.
- Haiku 4.5: all four cells, 128 episodes.
- Fable 5: stopped after 42 of its 128 episodes (the no-note cell and 10 of 32 note
  sessions) when the API account ran out of credit. By the plan's rule its incomplete
  cells are reported and not tested.
- GPT-5.5, Gemini and Qwen3.6-27B: not launched.
- Results and analysis output: `../results/response_v31/qx1_note_control/`.
- The run used the harness as run for R2 to R5, whose environment variables start with
  `IBENCH_`. The copy in `../src` and `../src2` is the same code with neutral names, so
  with it the launcher's `IBENCH_REPO` and the harness's `BENCH_...` variables apply.
