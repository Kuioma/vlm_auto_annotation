# 通用有序闭集动作批量标注入口

`wa2_frame_index_batch.yaml` 把批量调度与单条处理解耦：source 插件只负责发现
数据，processor 插件只负责处理一个 item，runner 只负责选择、重试、断点续跑和
聚合。更换处理脚本、模型、采样率或整个 processor 插件时，不需要修改 runner。

processor 从 `task.yaml` 读取任意长度的有序闭集动作。每个动作的详细定义、边界
规则和正反视觉线索来自 `ontology.yaml`；模型不得新增、删除或重排动作。它先在
全视频多视角低帧率接触图中，为 N 个动作定位 N-1 个粗边界，再分别用高帧率
局部窗口细化每个边界。边界 ID 使用该边界后开始的 `step_id`，最终 `segments`
始终按半开区间连续覆盖整条时间轴：
`A1=[0,b1)`、`A2=[b1,b2)`、...、`An=[b(n-1),duration_ms)`。
本批量配置显式传入 `--input-mode wa2`，保留数据集元数据和动作掩码选视角行为。单条普通视频默认使用 `direct` 模式，不要求这些元数据；详见根目录 [README](../../README.md)。

WA2 的 `pick_up → transfer → place` 只是当前 `task.yaml` 的一个配置实例；粗边界、
细边界及其位移会保留在输出中供审计。

先做无副作用检查：

```bash
uv run auto-annotate batch \
  --config examples/vllm_smoke/wa2_frame_index_batch.yaml \
  --plan-only
```

确认 vLLM 和 media staging 目录可用后启动正式标注：

```bash
uv run auto-annotate batch \
  --config examples/vllm_smoke/wa2_frame_index_batch.yaml
```

实际执行会在 stderr 持续显示 item、attempt 和成功/失败状态；自动化调用可加
`--quiet`。外部命令还可在 `argv` 中使用 `{item_json}`，读取 runner 为当前
attempt 原子生成的完整 BatchItem 快照。

配置分为四层：

- `source`：数据集位置、episode 元数据和多视角字段映射。
- `processor`：处理插件或命令、任务/本体、模型参数、粗细采样参数和处理身份文件。
- `execution`：断点续跑、重试、失败隔离和本次处理范围。
- `output`：批量状态与聚合结果的 workspace。

每个 runner 行为版本与处理身份的组合对应独立的 `<workspace>/<run_id>/`。处理参数、
`semantic_version`、共享 `identity_files` 内容或 runner 行为版本变化会生成新 run；
因此升级 runner 版本可能不会续用旧 namespace 的成功状态。共享身份不变时，单个
episode 的输入指纹变化只重跑该 episode。相同身份和输入下再次执行会跳过已经原子
提交成功的 item。

`execution.concurrency` 必须是正整数，表示最多同时处理多少个彼此独立的 episode。
并发只重叠各 item 的处理过程；每个 item 仍保留独立的尝试预算、重试记录、日志和
原子结果提交。`continue_on_error: false` 时，首个 item 用尽本次可用重试并最终失败
后，runner 不再分配新 item，但会让已经在途的 item 安全结束。对示例中的 27B
多模态模型，建议从 `2` 开始，再根据 GPU 显存占用和实际延迟逐步调整；可用并发度
取决于模型与服务配置，这里不作资源或吞吐保证。

runner 在整批中复用一个 processor session，因此自定义 processor 必须支持并发调用，
或在插件内部自行串行化不可共享的状态；收到取消时还必须清理自有资源后再传播取消。
内置 command processor 的框架状态可并发，并会在超时或取消时终止当前进程组，但
外部脚本仍应只把产物写入每个 item 独立的 `{output_dir}`，不要在共享 `{workspace}`
或 `cwd` 中写固定文件，也不要在主进程退出后留下继续写产物的后台进程。同一 run
还有进程级独占锁；不要启动多个相同 batch CLI 来叠加并发，应调整单次调用的
`execution.concurrency`。

主要产物：

- `output.jsonl`：当前数据集和当前输入指纹下的成功标注聚合。
- `output_index.jsonl`：输出行号到 item ID、dataset ID、输入指纹和记录哈希的映射。
- `failures.jsonl`：当前数据集中的最终失败项。
- `summary.json`：本次 selected/succeeded/skipped/failed/pending 统计。
- `invocations.json`：每次正式调用的配置哈希、执行范围和历史 summary；异常中断的
  调用可能保留为没有 summary 的 `running` 记录。
- `items/<item_id>/<input_fingerprint>/attempts/`：逐次输出、stdout/stderr 和失败信息。
- `identity.json`、`catalog.json`：处理身份和本次发现的数据目录。

`max_attempts` 是同一输入指纹的持久化尝试预算。预算耗尽后，可以在配置中提高
该值继续尝试；若还要重新考虑已判定为不可重试的失败，同时设置
`retry_terminal_failures: true`。这类执行策略调整不会改变 run identity。

第一次接真实模型时，建议先在 `execution` 中启用 `limit: 1` 做 canary；确认产物
后删除该限制继续全量，已成功的 canary 会被正常跳过。
