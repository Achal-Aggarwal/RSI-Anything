# Code-repair task

This task supplies program execution, candidate generation, world-model training and evaluation for the shared RSI workflow.

From the repository root:

```bash
rsi plan --config configs/code_repair/default.json
rsi run --config configs/code_repair/default.json
```

Prepare the environment using the [reproduction guide](../../docs/reproduction.en.md). Download fixed evaluation assets with `rsi assets code_repair --profile evaluate`; use `--profile full` for training. Sizes and offline preparation are in the [asset guide](../../docs/assets.md). Task-native commands live in `reproduce.py`; model configuration is in `config/experiment.json`. The four stages inherit checkpoints in order: `initial`, `semantic`, `expanded`, `incumbent`.

Run the task tests in its environment with `python -m unittest discover -s tasks/code_repair/tests -v` from the repository root.
