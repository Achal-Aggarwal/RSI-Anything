# Laboratory Planning

**Turn experimental objectives into executable laboratory plans with RSI-Anything.**

[简体中文](README.zh-CN.md) · [RSI-Anything](../../README.md) · [Hugging Face resources](https://huggingface.co/datasets/Gen-Verse/RSI-Anything/tree/main/laboratory_planning)

LabHorizon Level 2 asks an agent to plan a sequence of laboratory operations from an experimental context, a goal, available materials, constraints and an action library. A plan specifies which operations to perform, their parameters and how intermediate products pass between steps.

JEPA-Anything guides the choice of laboratory plans. A planning agent proposes four candidates, and the world model forecasts their quality, improvement, risk and uncertainty to select the next plan.

## Results

| Task | Evaluation size | Paper result |
| :--- | ---: | ---: |
| LabHorizon F200 · Level 2 | 200 tasks | **46.01%** |

## Setup

Use Python 3.10 or newer. Run commands from the repository root:

```bash
python -m pip install -r tasks/laboratory_planning/requirements.txt
export MODEL_DIR="$PWD/models/laboratory_planning"
```

| Inference component | Backbone or architecture | Adapter or checkpoint |
| :--- | :--- | :--- |
| JEPA candidate generation | [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | P4 for seed 1401, 1402 or 1403 |
| JEPA world model | JEPA-Anything | [Hugging Face checkpoints](https://huggingface.co/datasets/Gen-Verse/RSI-Anything/tree/main/laboratory_planning/checkpoints) |

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
| [generator.py](src/lab_inference/generator.py) | Candidate plan generation |
| [world.py](src/lab_inference/world.py) | JEPA forecasts and candidate selection |
| [features.py](src/lab_inference/features.py) | Task text and plan-structure features |
| [postprocess.py](src/lab_inference/postprocess.py) | Candidate plan normalization |

JEPA-Anything components retain [Apache-2.0](licenses/JEPA-Anything.LICENSE); the LabHorizon plan parser retains [MIT](licenses/LabHorizon-parser.LICENSE).
