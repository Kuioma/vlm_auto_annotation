# DashScope Qwen 视觉模型后端

主项目的标注管线通过兼容 OpenAI 的 Chat Completions 端点，支持阿里云百炼
（DashScope）的 Qwen 视觉模型。端点必须与 API key 所属的地域和工作空间匹配；
北京、新加坡工作空间的 URL 与美国地域端点不同。请将密钥保存在
`DASHSCOPE_API_KEY` 等环境变量中，不要将密钥值写入 YAML 文件。
端点也可以通过环境变量提供：

```bash
export DASHSCOPE_BASE_URL="https://workspace-id.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="your-key"
```

```yaml
backend:
  kind: openai_compatible
  base_url_env: DASHSCOPE_BASE_URL
  model_id: qwen3-vl-plus
  media_staging_root: staging
  api_key_env: DASHSCOPE_API_KEY
  # model_revision 可选，默认使用 model_id
  timeout_s: 180
  max_completion_tokens: 4096
  boundary_max_completion_tokens: 512
  temperature: 0
```

`base_url_env` 和直接填写 URL 的 `base_url` 必须且只能配置一个。
URL 环境变量缺失、为空或格式无效时，会在发送任何 HTTP 请求或持久化运行产物之前报错。
使用环境变量时，YAML 和规范化配置中只保留变量名；解析后的端点仍会记录在运行身份和
溯源信息（provenance）中，因此更改端点会产生不同的 run ID。

如果希望直接填写端点，请将 `base_url_env` 替换为：

```yaml
base_url: https://workspace-id.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

适配器每次生成发送一个非流式 `POST .../chat/completions` 请求，并启用 JSON Mode
（`response_format: {type: json_object}`）。完整的动态响应 Schema、Ontology 闭集词表、
局部毫秒时间规则和证据规则都会嵌入确定性构建的提示词中。响应仍需在本地通过
JSON Schema 校验和严格的领域模型校验。适配器不会调用 `/models` 或公共健康检查接口；
首次生成请求会验证端点、TLS、密钥、工作空间、地域、模型和协议是否可用。

媒体按照现有的源视频 PTS 采样计划进行采样。`run` 的粗标注和边界精修阶段分别使用
各自的局部采样计划，将一张 JPEG 接触图（采样帧拼图）编码为
`data:image/jpeg;base64,...` 图片。有序动作处理保留现有的一至三个视角布局，
但将合并后的布局作为一张图片发送；不提供通过多图或分页绕过限制的对外接口。
每张接触图最多包含 40 个时间单元，超出时需要降低粗标注或精修的采样帧率，或缩短窗口。
每个完整 data URI 的大小不得超过 10 MiB。请求结束后会删除临时媒体。

首次粗标注生成涉及的本地输入、Schema 和响应校验，均在创建持久化运行产物之前完成。
因此，首次生成失败不会留下采样计划、粗标注产物、最终 JSONL 或媒体工作目录。
粗标注成功后，如果后续边界精修失败，可能保留已经成功的粗标注产物，但不会生成成功的
最终输出。HTTP 错误只报告本地获取的 HTTP 状态，不会回显服务端返回的凭证、提示词
或媒体内容；适配器本身不执行重试。Mock 和 vLLM 后端保持原有的请求与溯源行为。

有序动作入口也支持同样的后端选择：

```bash
uv run auto-annotate-ordered \
  --backend-kind openai_compatible \
  --base-url-env DASHSCOPE_BASE_URL \
  --model-id qwen3-vl-plus \
  --api-key-env DASHSCOPE_API_KEY \
  ...
```

对于此后端，`--base-url` 与 `--base-url-env` 互斥。
`--prepare-only` 不会读取端点或 API key 环境变量，也不会发送 HTTP 请求。
