# 实验室规划

**使用 RSI-Anything，将实验目标转化为可执行的实验操作方案。**

[English](README.md) · [RSI-Anything](../../README.zh-CN.md) · [JEPA 世界模型](https://huggingface.co/datasets/Gen-Verse/RSI-Anything/tree/main/laboratory_planning/checkpoints)

LabHorizon Level 2 要求 Agent 根据实验背景、目标、现有材料、约束和操作库，规划完整的实验步骤。方案需要明确操作顺序、参数，以及中间产物在不同步骤之间的传递关系。

本模块提供最终规划策略的推理，以及 JEPA-Anything 世界模型引导的方案选择。规划策略提出候选方案，世界模型预测质量、改进收益、风险和不确定性，进而选择下一步方案。

## 实验结果

最终 **M_SFT100_E2** 规划策略在 LabHorizon F200 的 200 道任务上达到 **46.01%**。

| 评价设置 | 最终策略 | 动作序列相似度 | 参数准确度 | 最终得分 |
| :--- | :--- | ---: | ---: | ---: |
| LabHorizon F200 · Level 2 · 200 道任务 | M_SFT100_E2 | 45.79% | 46.23% | **46.01%** |

最终得分为动作序列相似度与参数准确度的平均值。最终策略使用贪心生成、操作签名修正，并将方案限制为最多 13 个操作调用。

## 环境准备

使用 Python 3.10 或以上版本，从仓库根目录运行：

```bash
python -m pip install -r tasks/laboratory_planning/requirements.txt
export MODEL_DIR="$PWD/models/laboratory_planning"
```

| 推理组件 | 基座或 checkpoint | 策略 adapter |
| :--- | :--- | :--- |
| 最终规划策略 | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | M_SFT100_E2 |
| JEPA 候选生成 | [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | 种子 1401、1402、1403 对应的 P4 |
| JEPA 世界模型 | [世界模型 checkpoint](https://huggingface.co/datasets/Gen-Verse/RSI-Anything/tree/main/laboratory_planning/checkpoints) | — |

通过 `--adapter` 指定本地策略 adapter 目录。下载世界模型：

```bash
hf download Gen-Verse/RSI-Anything --repo-type dataset \
  --include "laboratory_planning/checkpoints/*" \
  --local-dir "$MODEL_DIR/rsi-anything"
```

## 任务输入

使用 JSONL，每行一项任务：

| 字段 | 内容 |
| :--- | :--- |
| `id` | 任务标识 |
| `context` | 实验背景 |
| `goal` | 实验目标 |
| `constraints` | 约束列表 |
| `available_inputs` | 描述现有材料的 JSON 编码字符串 |
| `action_pool` | 可用实验操作的 Python 函数定义 |

[任务示例](examples/tasks.jsonl) 展示了输入格式。输出方案以关键字参数调用提供的操作，并通过命名变量传递中间产物。

## 最终规划策略推理

使用 M_SFT100_E2 生成最终方案：

```bash
python tasks/laboratory_planning/infer.py policy \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter "$MODEL_DIR/M_SFT100_E2" \
  --input tasks.jsonl \
  --output outputs/final_plans.jsonl \
  --device cuda:0
```

输出包含任务标识、原始模型回复、最终 `completion` 和后处理信息。

## 世界模型引导的规划

使用 P4 规划策略生成四个候选，再由对应的 JEPA 世界模型选择方案：

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

策略生成一个贪心候选和三个采样候选。JEPA 选择综合考虑预测质量、改进收益、风险、不确定性和方案结构合法性。推理期间，策略模型和世界模型的权重保持冻结。

种子 1402、1403 使用各自对应的 adapter、世界模型和种子参数。

## 对已有方案进行选择

输入每行包含 `task` 和四个 `candidates`，按 `a1`、`a2`、`a3`、`a4` 排列。每个候选包含 `candidate_id` 和 `completion`，其中 `a1` 为参照方案。格式见[候选示例](examples/candidates.jsonl)。

```bash
python tasks/laboratory_planning/infer.py select \
  --world-dir "$MODEL_DIR/rsi-anything/laboratory_planning/checkpoints/seed1401" \
  --input tasks/laboratory_planning/examples/candidates.jsonl \
  --output outputs/selected_plans.jsonl
```

方案选择在 CPU 上运行。输出包含选中的候选、最终 `completion`、全部候选，以及质量、收益、风险、不确定性和合法性的 `forecast`。添加 `--include-factors` 可输出 JEPA 潜在坐标。

## 代码结构

| 文件 | 作用 |
| :--- | :--- |
| [infer.py](infer.py) | 推理命令行入口 |
| [policy.py](src/lab_inference/policy.py) | 冻结策略加载与方案生成 |
| [world.py](src/lab_inference/world.py) | JEPA 预测与候选选择 |
| [features.py](src/lab_inference/features.py) | 任务文本和方案结构特征 |
| [postprocess.py](src/lab_inference/postprocess.py) | 操作签名修正与方案长度控制 |

JEPA-Anything 组件保留 [Apache-2.0](licenses/JEPA-Anything.LICENSE)，LabHorizon 方案解析器保留 [MIT](licenses/LabHorizon-parser.LICENSE)。
