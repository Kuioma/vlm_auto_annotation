# Auto Annotation

面向视频子任务的自动标注工具：从本地视频中生成结构化动作片段，通过粗标注和边界精修确定时间范围，导出 JSONL，并保留中间产物以便审计和复现。

当前工作区支持脚本化 mock、本地 vLLM 和 DashScope Qwen 视觉模型，提供单次标注、固定动作顺序标注、批量处理和离线评估入口。

## 功能与流程

- **通用标注**：读取 YAML 配置和 JSONL manifest，基于视频 PTS 采样，执行粗标注、边界精修和时间轴整理。
- **有序动作标注**：由 `task.yaml` 指定动作顺序，结合 Ontology 中的动作定义，用多视角接触图定位并细化动作边界。
- **批量处理**：通过 source / processor 插件发现与处理数据，支持并发、重试、失败隔离和断点续跑。
- **离线评估**：对照人工 Gold 标注计算边界误差、时间 IoU 等指标，生成报告和人工复核队列。
- **数据约束与溯源**：Schema 定义字段，Ontology 定义词表；模型响应经过 JSON Schema 和严格 Pydantic 校验，结果记录配置与输入身份。

时间统一使用视频内绝对整数毫秒，片段采用半开区间 `[start_ms, end_ms)`。通用标注允许片段间有间隙，最终片段有序且不重叠；有序动作入口按既定动作顺序连续覆盖整条时间轴。

## 环境准备

需要 Python **3.12 或 3.13**、`uv`，以及已加入 `PATH` 的 `ffmpeg` 和 `ffprobe`。真实模型推理还需要可访问的模型服务；安装本项目不会启动 vLLM 服务。

在仓库根目录执行：

```bash
uv sync --extra test
uv run auto-annotate --help
uv run auto-annotate-ordered --help
```

## 快速开始：无需模型服务

下面生成一段 1 秒的黑色视频，并使用仓库中的 mock 响应运行完整流程。mock 返回预设标注，用于验证管线，不代表视频识别效果。所有示例产物位于已被 Git 忽略的 `artifacts/quickstart/`。

```bash
mkdir -p artifacts/quickstart

ffmpeg -y -v error -f lavfi \
  -i 'color=c=black:s=320x240:d=1:r=30' \
  -pix_fmt yuv420p artifacts/quickstart/video.mp4

cat > artifacts/quickstart/input.jsonl <<'EOF'
{"video_id":"video-1","video_uri":"video.mp4","dataset_id":"demo","schema_version":"subtask-v1","ontology_id":"kitchen-v1"}
EOF

cat > artifacts/quickstart/run.yaml <<'EOF'
schema_path: ../../tests/fixtures/schema/subtask-v1.yaml
ontology_path: ../../tests/fixtures/ontology/kitchen-v1.yaml
manifest_path: input.jsonl
artifact_root: stages
output_path: output.jsonl
backend:
  kind: mock
  fixture_path: ../../tests/fixtures/mock/annotation.json
sampling:
  coarse_fps: 2.0
  refine_fps: 6.0
  refine_window_ms: 4000
timeline:
  merge_gap_ms: 250
  min_segment_ms: 300
EOF

uv run auto-annotate run --config artifacts/quickstart/run.yaml
cat artifacts/quickstart/output.jsonl
```

预期输出包含任务描述“将杯子放到托盘”，以及 `atomic_action: grasp`、时间范围 `[100, 800)` 的片段。再次执行相同配置和输入，mock 流程应生成相同的输出字节。

### 输入约定

| 文件 | 内容 |
| --- | --- |
| `run.yaml` | Schema、Ontology、manifest、输出路径、后端与采样配置 |
| `input.jsonl` | 每行一个视频，包含 `video_id`、`video_uri`、`dataset_id`、`schema_version`、`ontology_id` |
| Schema YAML | 片段字段、类型、必填项和文本输出约束 |
| Ontology YAML | 动作、交互对象等词表及语义定义 |
| `task.yaml` | 有序动作入口使用的固定步骤及对应字段值 |

`run.yaml` 中的相对路径以配置文件目录为基准；manifest 的 `video_uri` 以 manifest 所在目录为基准，必须指向存在的本地文件。一次 `run` 必须包含同一数据集的非空记录，视频 ID 唯一且可安全用于路径，Schema / Ontology ID 与载入的契约一致。

mock 响应按请求 ID 排队，示例 fixture 绑定 `video-1`。更换视频 ID 或片段结构时，需要同步修改 fixture。

### 输出结构

```text
artifacts/quickstart/
├── output.jsonl
└── stages/
    └── run-<digest>/
        └── video-1/
            ├── coarse-sample-plan.json
            ├── coarse.json
            ├── refined.json
            └── final.json
```

最终结果包含时间片段、结构化字段、文本描述、质量信号和 provenance。通用入口的 JSONL 按 `video_id` 排序并原子写入；输入、配置和行为版本参与运行身份计算。真实模型的生成结果仍可能存在波动。

## 接入真实模型

| 后端 | 配置方式 | 示例 / 说明 |
| --- | --- | --- |
| Mock | `backend.kind: mock`，指定 `fixture_path` | 上面的快速开始 |
| vLLM | `backend.kind: vllm`，配置服务地址、模型 ID / revision、vLLM 版本与媒体暂存目录 | [有序动作批量示例](examples/vllm_smoke/README.md) |
| DashScope | `backend.kind: openai_compatible`，通过环境变量提供端点与 API key | [接入说明](docs/dashscope-openai-compatible.md)、[run 配置](examples/dashscope_openai_compatible/run.yaml) |

`openai_compatible` 当前采用 DashScope Qwen vision 协议配置，并非对所有兼容服务的通用适配。示例 run 配置需要在同目录准备自己的 `input.jsonl`，再执行：

```bash
uv run auto-annotate run --config examples/dashscope_openai_compatible/run.yaml
```

有序动作入口支持一至三个视角。任务和词表样例位于 [pick_and_place_bolt](examples/pick_and_place_bolt/)，完整参数可通过 `uv run auto-annotate-ordered --help` 查看。

## 批量标注

批量配置分为 `source`、`processor`、`execution` 和 `output` 四部分。内置 WA2 source 读取 episode 元数据，command processor 调用独立处理命令。

仓库示例含开发环境的数据集绝对路径、模型服务地址、媒体暂存目录和输出 workspace。运行前请修改这些值，以及 `identity_files` 中的数据文件路径。

```bash
# 验证配置与输入，查看本次选中的处理计划
uv run auto-annotate batch \
  --config examples/pick_and_place_bolt/batch.yaml --plan-only

# 正式处理
uv run auto-annotate batch \
  --config examples/pick_and_place_bolt/batch.yaml
```

首次运行可设置 `execution.limit: 1` 验证单条数据，再去掉限制扩大范围。`execution.concurrency` 控制并发数，应根据服务容量调整。相同运行身份和输入下，启用 `resume` 会跳过已提交成功的 item；`max_attempts` 是同一输入指纹的持久化尝试预算。

批量产物写入 `<workspace>/<run_id>/`，包括成功结果 `output.jsonl`、行索引 `output_index.jsonl`、失败项 `failures.jsonl`、统计 `summary.json` 和逐次尝试日志。更多续跑和插件约定见[批量入口说明](examples/vllm_smoke/README.md)。

## 离线评估

准备人工 Gold 集、标注目录和预测 JSONL，修改[评估配置示例](examples/evaluation/wa2_pick_place_boundary_v1.yaml)中的输入、输出路径后运行：

```bash
uv run auto-annotate evaluate \
  --config examples/evaluation/wa2_pick_place_boundary_v1.yaml
```

该示例启用边界误差、时间 IoU 和 episode bootstrap 置信区间，生成机器可读结果、Markdown 报告及复核队列。固定动作顺序任务中，标签和动作序列由任务配置给定，示例关闭这些指标，避免把配置约束当作识别能力。

## 开发与测试

```bash
# 全量测试
uv run pytest -q

# 无模型服务的真实视频端到端测试
uv run pytest tests/integration/test_cli_mock_e2e.py::test_cli_runs_mock_pipeline_on_real_video -q

# 流水线单元测试
uv run pytest tests/pipeline -q
```

真实视频集成测试在缺少 FFmpeg / FFprobe 时会跳过。仓库目前未配置统一的 lint、format 或静态类型检查命令。

| 目录 | 职责 |
| --- | --- |
| `src/auto_annotation/config/`、`domain/` | 配置与领域模型 |
| `src/auto_annotation/media/` | 视频探测、PTS 采样与媒体物化 |
| `src/auto_annotation/ontology/`、`prompts/` | 标注契约、词表与提示词 |
| `src/auto_annotation/inference/`、`pipeline/` | 后端适配和标注阶段 |
| `src/auto_annotation/batch/`、`evaluation/` | 批量调度与离线评估 |
| `src/auto_annotation/artifacts/`、`exporters/` | 中间产物和结果导出 |
| `tests/`、`examples/` | 测试与运行配置示例 |
| `thirdparty/vllm-main/` | 独立的 vLLM 源码快照 |

## 当前限制

- 通用 `auto-annotate run` 只接受单个短视频 chunk，上限为 **120,000 ms**；尚未实现长视频多 chunk 合并。
- 当前结果导出使用 JSONL，未提供 Parquet 导出。
- DashScope 接触图最多包含 40 个时间单元，完整图片 data URI 上限为 10 MiB；超限时需降低采样率或缩短窗口。
- 单次 `run` 没有通用自动重试层；批量入口按配置对 item 进行重试。
- 语义质量评估和 Gold 标注一致性评估尚未实现。
