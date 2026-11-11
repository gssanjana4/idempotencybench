# QX1 plan: content-free note control (written before any QX1 episode)

Question. The two-sentence note ("a previous session was working on this task and was interrupted") removed almost all duplicates on restart. Is that because of what the note says, or because any NOTE line makes a model more careful?

Design. Same harness, graphs, interrupt points and limits as R2/R3 Part A. 16 graphs x 2 samples per cell, all four cells of a model in one run:
- amnesiac: task only
- bare_hint: "NOTE: a previous session was working on this task and was interrupted. Finish the task."
- note_null: "NOTE: Finish the task."
- note_irrelevant: "NOTE: this task was assigned to you from the operations queue. Finish the task."

Models: Haiku 4.5, Fable 5, GPT-5.5, Gemini, Qwen3.6-27B (Sonnet 5 is left out: its no-note baseline was 6/32, so there is nothing to remove). 640 episodes.

Outcome. Graph IVR: share of resumed sessions that repeat a committed step. Secondary: sessions that call list_effects before any effectful call; completion; the same counts with session-1 keys honoured.

Tests, per model. Paired sign-flip permutation over the 16 graphs (10,000 flips, seed 20261025), one-sided, Holm over the two control notes:
- H1: gIVR under note_null is higher than under bare_hint.
- H2: gIVR under note_irrelevant is higher than under bare_hint.
Descriptive: each control against amnesiac, two-sided.

Reading, fixed now.
- "Information effect" is supported for a model if H1 and H2 both hold.
- If a control note is not above bare_hint and is below amnesiac, the note effect for that model is (at least partly) the salience of a NOTE line, and the paper will say so in Sections 5 and 9.
- A cell under 90% of its planned episodes is reported but not tested. Provider refusals are excluded and counted.

Limits known in advance. Two plan shapes, so significance is over graphs only. One wording per control. Not run on Sonnet 5.

Budget. Hard caps in the config: 13.00 USD Anthropic + 8.50 USD OpenRouter. The run stops before a cap is passed.

Launch note (added 11 Nov 2026, 02:06 UTC, before any QX1 episode). The two Claude models are launched first, from `qx1_note_control_claude_only.json`: the same cells, texts, samples, limits and output folder as `qx1_note_control_cap21.json`, restricted to its Claude group (256 episodes, cap 13.00 USD). The OpenRouter group (GPT-5.5, Gemini, Qwen3.6-27B; 384 episodes, cap 8.50 USD) is launched later with the cap21 file, if at all. Each model's four cells still come from one run. Tests and readings are per model, so nothing above changes. This file, both launch files and the launcher were pushed to the public repository before the first episode.
