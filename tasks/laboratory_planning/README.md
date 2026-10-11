# Laboratory Planning

**Turn experimental objectives into executable laboratory plans with RSI-Anything.**

[简体中文](README.zh-CN.md) · [RSI-Anything](../../README.md) · [JEPA checkpoints](https://huggingface.co/datasets/Gen-Verse/RSI-Anything/tree/main/laboratory_planning/checkpoints)

LabHorizon Level 2 asks an agent to plan a sequence of laboratory operations from an experimental context, a goal, available materials, constraints and an action library. A plan specifies which operations to perform, their parameters and how intermediate products pass between steps.

This module provides frozen planning-policy inference and JEPA-Anything world-model selection. The planning policy proposes candidate plans; the world model predicts their quality, improvement, risk and uncertainty, then selects the next plan.

## Results

The final **M_SFT100_E2** planning policy reaches **46.01%** on the 200-task LabHorizon F200 evaluation.

| Evaluation | Final policy | Action sequence similarity | Parameter accuracy | Final score |
| :--- | :--- | ---: | ---: | ---: |
| LabHorizon F200 · Level 2 · 200 tasks | M_SFT100_E2 | 45.79% | 46.23% | **46.01%** |

The final score is the mean of action sequence similarity and parameter accuracy. Final-policy inference uses greedy generation, action-signature repair and a limit of 13 action calls.

## Setup

Use Python 3.10 or newer. Run commands from the repository root:

```bash
python -m pip install -r tasks/laboratory_planning/requirements.txt
export MODEL_DIR="$PWD/models/laboratory_planning"
```

| Inference component | Backbone or checkpoint | Adapter |
| :--- | :--- | :--- |
| Final planning policy | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | M_SFT100_E2 |
| JEPA candidate generation | [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | P4 for seed 1401, 1402 or 1403 |
| JEPA world model | [World checkpoints](https://huggingface.co/datasets/Gen-Verse/RSI-Anything/tree/main/laboratory_planning/checkpoints) | — |

Pass the local policy adapter directory with `--adapter`. Download the world checkpoints with:

```bash
hf download Gen-Verse/RSI-Anything --repo-type dataset \
  --include "laboratory_planning/checkpoints/*" \
  --local-dir "$MODEL_DIR/rsi-anything"
```

## Task inputs

Use JSONL with one task per line:

| Field | Content |
| :--- | :--- |
| `id` | Task identifier |
| `context` | Experimental background |
| `goal` | Requested experimental outcome |
| `constraints` | List of constraints |
| `available_inputs` | JSON-encoded string describing named materials |
| `action_pool` | Python definitions of the available operations |

The [task example](examples/tasks.jsonl) illustrates the input format. Plans use Python calls to the supplied operations, with keyword arguments and named intermediate products.

## Final planning policy

Generate final plans with M_SFT100_E2:

```bash
python tasks/laboratory_planning/infer.py policy \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter "$MODEL_DIR/M_SFT100_E2" \
  --input tasks.jsonl \
  --output outputs/final_plans.jsonl \
  --device cuda:0
```

Each output contains the task identifier, raw model response, final `completion` and postprocessing details.

## Planning with the JEPA world model

Generate four candidates with the P4 planning policy and select a plan with its matching JEPA checkpoint:

```bash
python tasks/laboratory_planning/infer.py world \
  --base-model Qwen/Qwen3.5-4B-Base \
  --adapter "$MODEL_DIR/seed1401/P4" \
  --world-dir "$MODEL_DIR/rsi-anything/laboratory_planning/checkpoints/seed1401" \
  --seed 1401 \
  --input tasks.jsonl \
  --output outputs/jepa_plans.jsonl \
  --device cuda:0
```

The policy produces one greedy candidate and three sampled candidates. JEPA selection combines predicted quality and improvement with risk, uncertainty and structural validity. The planning policy and world model remain frozen during inference.

Use the corresponding adapter, world checkpoint and seed for 1402 or 1403.

## Select existing candidate plans

Provide JSONL records containing `task` and four `candidates`, ordered `a1`, `a2`, `a3`, `a4`. Each candidate has a `candidate_id` and `completion`; `a1` is the reference plan. See the [candidate example](examples/candidates.jsonl).

```bash
python tasks/laboratory_planning/infer.py select \
  --world-dir "$MODEL_DIR/rsi-anything/laboratory_planning/checkpoints/seed1401" \
  --input tasks/laboratory_planning/examples/candidates.jsonl \
  --output outputs/selected_plans.jsonl
```

Candidate selection runs on CPU. Outputs include the selected candidate, final `completion`, all candidates and a `forecast` with predicted quality, improvement, risk, uncertainty and validity. Add `--include-factors` to return the JEPA latent coordinates.

## Code

| File | Purpose |
| :--- | :--- |
| [infer.py](infer.py) | Inference command line |
| [policy.py](src/lab_inference/policy.py) | Frozen policy loading and generation |
| [world.py](src/lab_inference/world.py) | JEPA forecasts and candidate selection |
| [features.py](src/lab_inference/features.py) | Task text and plan-structure features |
| [postprocess.py](src/lab_inference/postprocess.py) | Action-signature repair and plan length control |

JEPA-Anything components retain [Apache-2.0](licenses/JEPA-Anything.LICENSE); the LabHorizon plan parser retains [MIT](licenses/LabHorizon-parser.LICENSE).
