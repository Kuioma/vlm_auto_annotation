# Auto Annotation

本项目的主流程是 **用 VLM 对视频做有序动作时间分段**，运行入口为：

```bash
uv run auto-annotate-ordered --config examples/pick_and_place_bolt/ordered_api.yaml
```

你提供一条视频、动作定义和既定动作顺序，程序调用视觉语言模型找到动作之间的时间边界，输出每个动作的开始时间和结束时间。支持调用 DashScope Qwen API，也支持连接本地 vLLM 服务。

下面以“把螺丝从蓝色料箱搬到目标容器”为完整例子，依次说明如何准备数据、定义动作、配置模型、检查输入、运行并查看结果。批量运行和离线评估在单条流程之后介绍，通用标注与 mock 验证放在文档末尾。

## 1. 先明确：输入什么，模型判断什么，输出什么

假设一条视频记录了完整的抓放过程，我们希望分为三个连续阶段：

```text
抓取 pick_up → 搬运 transfer → 放置 place
```

这些动作及其顺序由你在 `task.yaml` 中指定。模型需要判断的是两个交界时刻：什么时候从抓取进入搬运，什么时候从搬运进入放置。

例如，一条视频长 10 秒，模型最终找到的两个边界分别为 3200 ms 和 7800 ms，结果就是：

| 动作 | 起点 | 终点 | 含义 |
| --- | --- | --- | --- |
| `pick_up` | 0 | 3200 | 从视频开始到稳定抓持建立 |
| `transfer` | 3200 | 7800 | 从稳定抓持建立到进入最终放置阶段 |
| `place` | 7800 | 10000 | 从开始最终对位、下放或释放到视频结束 |

这里的时间仅用于说明输出，不是仓库示例的真实预测。所有时间都是视频内绝对整数毫秒，区间采用 `[start_ms, end_ms)`；前一段终点就是后一段起点，最终分段连续覆盖整个视频。

程序会执行以下流程：

```text
加载运行配置和三个任务定义文件
              ↓
读取视频及数据集元数据，选择并检查相机视角
              ↓
全视频低帧率采样，制作带帧编号的接触图
              ↓
VLM 根据动作顺序和边界规则定位 N−1 个粗边界
              ↓
每个粗边界附近高帧率采样，分别调用 VLM 精修
              ↓
将边界转换为 N 个连续片段，写入 output.jsonl
```

“接触图”是一张由多个采样时刻的画面拼成的图片；同一时刻可以包含头部和腕部视角。模型根据图中的帧编号选择边界，程序再映射回原视频时间。

**这条流程适用于动作顺序已知、每条视频都应包含完整步骤的任务。** 模型不会自动发现新动作，也不会自行删掉或重排步骤。当前步骤 ID 必须唯一；如果视频经常缺少某一步、顺序变化或重复同一动作，不能直接把当前三步模板当作通用动作发现器使用。

## 2. 准备环境和一条视频

### 安装与检查

需要 Python **3.12 或 3.13**、`uv`、`ffmpeg` 和 `ffprobe`。下面所有命令均在仓库根目录执行：

```bash
uv sync --extra test
ffmpeg -version
ffprobe -version
uv run auto-annotate-ordered --help
mkdir -p /tmp/auto_annotation_ordered_media
```

最后一条命令创建媒体暂存目录，采样图会在这里临时生成。后续配置中的 `backend.media_staging_root` 必须指向一个已存在的目录。

### 当前支持的数据布局

当前有序处理器会从视频路径定位 WA2 数据集根目录，并读取该数据集的元数据。请选择现有数据集中的一条 episode；它需要类似下面的布局：

```text
/path/to/dataset/
├── meta/
│   ├── episodes.jsonl
│   ├── modality.json
│   └── training_mask.json
└── videos/
    └── chunk-000/
        ├── observation.images.head_rgb/
        │   └── episode_000000.mp4
        └── observation.images.right_wrist_rgb/
            └── episode_000000.mp4
```

| 数据 | 在流程中的作用 |
| --- | --- |
| 头部视频 `video` | 必填的主视角；必须使用 `videos/chunk-*/<video-key>/<episode>.mp4` 路径布局 |
| `meta/episodes.jsonl` | 匹配这条头部视频所属的 episode，并找到同一 episode 的其他视角视频 |
| `meta/modality.json` | 将头部、左腕、右腕等逻辑视角映射到数据集的实际视频字段 |
| `meta/training_mask.json` | 根据启用的 `left_*`、`right_*` 动作决定是否使用对应腕部视角 |
| 左右腕视频 | 与头部视频共同提供视觉证据；选中的多视角视频需要通过时间同步检查 |

程序支持一至三个视角。通常可以让它从元数据自动寻找腕部视频，也可以用 `left_wrist_video`、`right_wrist_video` 显式指定。显式路径不能启用被动作掩码禁用的视角：例如数据集没有启用右侧动作，就应删除示例中的 `right_wrist_video`，否则会报错。

**当前不能只把一个任意 MP4 放进目录就运行。** 上述三个元数据文件需要存在，且视频路径和映射必须与其一致。当前也只接受最多 120 秒的单个视频 chunk；接触图采样数量还有额外限制，后面会解释如何调整采样率。

## 3. 准备四个 YAML：一个运行配置，三个任务定义

先使用仓库现有的 [pick_and_place_bolt](examples/pick_and_place_bolt/) 示例：

```text
examples/pick_and_place_bolt/
├── ordered_api.yaml   # 运行配置：API 后端、视频、采样和输出位置
├── ordered.yaml       # 另一份运行配置：本地 vLLM 后端
├── schema.yaml        # 定义每段输出需要哪些字段
├── ontology.yaml      # 定义字段可用值、动作含义和边界规则
└── task.yaml          # 定义本次任务及动作的先后顺序
```

一次单条运行选择 `ordered_api.yaml` 或 `ordered.yaml` 其中之一，由它引用另外三个文件：

```yaml
schema_path: schema.yaml
ontology_path: ontology.yaml
task_path: task.yaml
```

**这三个相对路径都从运行配置所在目录开始找。** 如果运行配置是 `examples/pick_and_place_bolt/ordered_api.yaml`，则读取同目录的 `schema.yaml`、`ontology.yaml` 和 `task.yaml`，与终端当前目录无关。将运行配置移动到新目录时，需要一起复制这三个文件，或者修改引用路径；也可以直接填写绝对路径。

三个文件共同决定模型要找什么以及结果如何组织。下面按“输出结构 → 动作语义 → 本次动作顺序”逐一说明。

### 3.1 schema.yaml：每个片段需要输出哪些字段

Schema 定义字段名称、类型、是否必填及枚举值来源。当前示例的完整内容是：

```yaml
schema_id: pick-and-place-bolt-v1
segment_values:
  atomic_action:
    type: enum
    source: ontology.atomic_actions
    required: true
  interaction_object:
    type: enum
    source: ontology.interaction_objects
    required: true
  target:
    type: string
    required: true
text_output:
  field: sentence
  language: zh-CN
  style: imperative
  required: true
```

逐项理解：

| 配置 | 含义 |
| --- | --- |
| `schema_id` | 这份字段规范的 ID，`task.yaml` 必须引用同一个 ID |
| `segment_values` | 每个输出片段的 `values` 对象需要包含哪些字段 |
| `atomic_action.type: enum` | 动作值必须来自指定词表，不能随意生成一个新名称 |
| `source: ontology.atomic_actions` | 动作词表来自 `ontology.yaml` 中的 `vocabularies.atomic_actions` |
| `interaction_object` | 交互对象，值来自 `ontology.interaction_objects` |
| `target.type: string` | 目标是字符串；当前 Schema 没有将它限制为枚举词表 |
| `required: true` | 对应字段必须存在 |
| `text_output` | 描述字段为 `sentence`，声明语言、风格和必填要求 |

因此，这个任务的一个片段会有类似下面的 `values`：

```json
{"atomic_action": "transfer", "interaction_object": "workpiece", "target": "destination_container"}
```

Schema 本身不指定动作顺序，也不在这里填写每段的时间。`start_ms`、`end_ms` 由分段程序根据模型选择的边界生成。在有序入口中，步骤的 `values` 和 `sentence` 来自 Task 配置；最终描述不是让模型自由重写。

### 3.2 ontology.yaml：每个动作是什么意思，边界怎么判断

Ontology 给 Schema 引用的词表提供实际值，并告诉模型这些动作在视觉上如何区分。当前示例包含三种动作和一种对象，其完整定义如下：

```yaml
ontology_id: pick-and-place-bolt-v1
revision: v1
vocabularies:
  atomic_actions:
    - id: pick_up
      name: 抓取阶段
      definition: >-
        从时间轴开始到夹爪首次对工件建立稳定控制的持续阶段。
      temporal_type: interval_action
      boundary_rule: >-
        选择能够回溯确认稳定抓持已经成立的最早采样帧。
        该边界结束 [timeline_start, pick_up) 抓取阶段，并开始搬运阶段。
        可以使用后续帧验证抓持是否稳定，但不得将事件时间推迟到工件被抬起或离开容器。
      positive_cues:
        - 夹爪已形成有效夹持
        - 后续画面可验证工件与夹爪保持同步运动
      negative_cues:
        - 夹爪仅接触工件
        - 夹爪正在闭合但尚未稳定控制工件
        - 工件已经被抬起或离开容器
        - 工件仅仅离开主相机画面
    - id: transfer
      name: 搬运
      definition: 保持对工件的稳定控制并将其移向目标位置的持续过程
      temporal_type: interval_action
      boundary_rule: >-
        开始边界选择稳定抓持已经成立、搬运开始的最早采样帧；
        结束边界选择工件进入目标落放区域、搬运转为最终放置的最早采样帧。
        搬运使用半开区间，两个边界分别属于搬运段和后续放置段的起点。
      positive_cues:
        - 工件在夹持中随夹爪运动
        - 夹爪与工件之间的控制关系持续保持
      negative_cues:
        - 稳定抓持尚未成立时的接近或闭合夹爪
        - 工件已经释放后的撤离动作
        - 仅凭工件离开主相机画面推断搬运结束
    - id: place
      name: 放置阶段
      definition: >-
        从工件进入目标落放区域、搬运转为最终对位和释放，到时间轴结束的持续阶段。
      temporal_type: interval_action
      boundary_rule: >-
        选择能够确认工件已经到达目标落放区域，并开始最终对位、下放或释放过程的最早采样帧。
        该边界结束 [pick_up, place) 搬运阶段，并开始 [place, timeline_end) 放置阶段。
        可以使用后续帧验证放置是否完成，但不得将边界推迟到释放完成或机械手撤离。
      positive_cues:
        - 工件已经进入目标容器或工作区域
        - 运动从跨区域搬运转为最终对位、下放或释放
      negative_cues:
        - 工件仍在向目标区域进行跨区域搬运
        - 仅凭工件离开主相机画面推断放置开始
        - 等到工件已经释放或机械手撤离后才选择边界
  interaction_objects:
    - id: workpiece
      name: 工件
      definition: 被机器人抓持、移动并释放到目标位置的待分拣物体
```

动作词条各字段的用途：

| 字段 | 作用 |
| --- | --- |
| `id` | 程序使用的动作标识，例如 `transfer`；Task 中的步骤引用它 |
| `name` | 人可读的名称，例如“搬运” |
| `definition` | 描述这个动作覆盖的行为范围 |
| `temporal_type` | 时间语义类型；当前三种动作使用 `interval_action`，表示持续一段时间的动作 |
| `boundary_rule` | 具体说明从哪个视觉时刻切换到下一阶段 |
| `positive_cues` | 支持边界判断的画面证据 |
| `negative_cues` | 不应当被误判为边界的情况 |

例如，“抓取完成”在本任务中意味着夹爪已经稳定控制工件。它可能早于工件被抬出料箱，因此不能简单把“离开料箱”当作抓取与搬运的边界。同样，“进入放置阶段”也不等于“已经完全松开夹爪”。这类业务判定规则要写在 Ontology 中，程序会将其带入模型提示词。

如果你发现边界总是偏早或偏晚，应先检查这里的规则是否与人工标注标准一致，再考虑调整采样参数。仅修改动作的中文名称不足以明确时间边界。

### 3.3 task.yaml：这次任务按什么顺序分段

Task 将 Schema、Ontology 和具体步骤组合起来。当前抓放螺丝任务的完整配置是：

```yaml
version: 1
task_id: pick-and-place-bolt-v1
schema_id: pick-and-place-bolt-v1
ontology_id: pick-and-place-bolt-v1
task_summary: 将螺丝从蓝色料箱转移至目标容器
steps:
  - step_id: pick_up
    values:
      atomic_action: pick_up
      interaction_object: workpiece
      target: destination_container
    sentence: 抓取阶段：从时间轴开始到夹爪首次对工件建立稳定控制的持续阶段。
  - step_id: transfer
    values:
      atomic_action: transfer
      interaction_object: workpiece
      target: destination_container
    sentence: 搬运：保持对工件的稳定控制并将其移向目标位置的持续过程。
  - step_id: place
    values:
      atomic_action: place
      interaction_object: workpiece
      target: destination_container
    sentence: 放置阶段：从工件进入目标落放区域、搬运转为最终对位和释放，到时间轴结束的持续阶段。
```

`steps` 是有序列表，程序按书写顺序处理：第一个是 `pick_up`，第二个是 `transfer`，第三个是 `place`。每个步骤同时给出该段最终输出使用的 `values` 和 `sentence`。

| 配置 | 含义 |
| --- | --- |
| `version` | Task 文件格式版本，当前为 `1` |
| `task_id` | 本次任务定义的标识 |
| `schema_id`、`ontology_id` | 引用另外两个文件中的对应 ID |
| `task_summary` | 整条视频应完成的任务描述 |
| `steps[].step_id` | 步骤 ID，必须对应 Ontology 的动作 ID |
| `steps[].values` | 这个步骤最终输出的结构化值，必须通过 Schema 和词表校验 |
| `steps[].sentence` | 这个步骤最终输出的描述 |

三个文件必须满足这些对应关系：

```text
Task.schema_id       = Schema.schema_id
Task.ontology_id     = Ontology.ontology_id
Task.step_id         = 该步 values.atomic_action
Task.step_id         ∈ Ontology.vocabularies.atomic_actions 的 ID
```

当前至少需要两个步骤，且步骤 ID 必须唯一；每个动作词条必须提供 `temporal_type` 和 `boundary_rule`。对于 N 个步骤，模型寻找 N−1 个内部边界。边界使用其后开始的步骤 ID 命名，所以本例两个边界 ID 是 `transfer` 和 `place`；没有额外的 `pick_up` 开始边界，因为第一段直接从时间轴起点开始。

### 3.4 改任务时，应该改哪份文件

| 你的需求 | 修改位置 |
| --- | --- |
| 换一条视频、换模型、调整采样率或输出目录 | 运行配置 `ordered_api.yaml` / `ordered.yaml` |
| 改“抓取 → 搬运 → 放置”的步骤顺序或每步输出内容 | `task.yaml` 的 `steps`，并确认边界规则仍适用 |
| 改“稳定抓持”的判定标准或放置开始的判定时刻 | `ontology.yaml` 的动作定义、边界规则和线索 |
| 新增一种动作 | 在 Ontology 中定义词条，再在 Task 中加入对应步骤 |
| 新增输出字段或修改字段类型 | 修改 Schema，并同步补齐 Task 的值；如果是枚举，还需更新 Ontology |

如果只是跑现有抓放任务，可以直接使用仓库中这三份文件，先改运行配置中的视频和服务参数。

## 4. 配置一次单条运行

### 4.1 使用 API：ordered_api.yaml

[完整 API 示例](examples/pick_and_place_bolt/ordered_api.yaml)如下。将两个 `/path/to/dataset/...` 改为你实际的同一 episode 视频路径；如果让元数据自动寻找腕部视频，可以省略 `right_wrist_video`。

```yaml
video: /path/to/dataset/videos/chunk-000/observation.images.head_rgb/episode_000000.mp4
right_wrist_video: /path/to/dataset/videos/chunk-000/observation.images.right_wrist_rgb/episode_000000.mp4
output_dir: ../../artifacts/ordered_api

schema_path: schema.yaml
ontology_path: ontology.yaml
task_path: task.yaml

backend:
  kind: openai_compatible
  base_url_env: DASHSCOPE_BASE_URL
  api_key_env: DASHSCOPE_API_KEY
  model_id: qwen3-vl-plus
  media_staging_root: /tmp/auto_annotation_ordered_media
  timeout_s: 180

sampling:
  coarse_fps: 2
  refine_fps: 10
  refine_window_ms: 2000
columns: 3
```

`backend.kind: openai_compatible` 选择现有的 DashScope Qwen vision 适配器。这里 `base_url_env`、`api_key_env` 填的是**环境变量名称**，不是地址或密钥本身。请在运行命令所在的终端设置实际值：

```bash
# 替换为与你的地域和工作空间匹配的实际端点与密钥
export DASHSCOPE_BASE_URL='https://你的服务域名/compatible-mode/v1'
export DASHSCOPE_API_KEY='你的 API key'
```

也可以把 `base_url_env` 一行替换为 `base_url: https://.../compatible-mode/v1` 直接填写地址，二者只能选择一个。API 的 `model_revision` 可选，未填写时使用 `model_id`。`timeout_s` 是每次模型请求的超时秒数，不是整条视频的总运行时限。

API 模式把接触图作为图片数据随请求发送，不需要服务端访问本地视频路径。该后端使用 DashScope Qwen vision 协议，并不保证任意 OpenAI 兼容服务都可直接使用；详细协议与限制见 [API 接入说明](docs/dashscope-openai-compatible.md)。

### 4.2 使用本地 vLLM：ordered.yaml

如果你已经部署了本地模型服务，使用 [本地 vLLM 示例](examples/pick_and_place_bolt/ordered.yaml)。它的视频、Schema、Ontology、Task 和采样配置与 API 方式相同，后端部分改为：

```yaml
backend:
  kind: vllm
  base_url: http://127.0.0.1:8000/v1
  model_id: Qwen/Qwen3.6-27B
  model_revision: 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
  vllm_version: 0.24.0
  media_staging_root: /tmp/auto_annotation_ordered_media
  timeout_s: 600
```

以上模型名称、revision、服务版本都是仓库示例值，需要与你实际部署的服务保持一致。它们用于请求和运行记录，不会替你安装或切换模型。请先启动 vLLM 服务，并确保服务可访问媒体暂存目录、允许读取对应的本地媒体路径；本项目的安装和标注命令不会启动 vLLM 服务。

### 4.3 采样参数怎么选

| 参数 | 示例值 | 作用 |
| --- | --- | --- |
| `sampling.coarse_fps` | `2` | 全视频每秒约采样两帧，供模型粗定位动作边界 |
| `sampling.refine_fps` | `10` | 每个边界附近每秒约采样十帧，必须大于粗采样率 |
| `sampling.refine_window_ms` | `2000` | 以粗边界为中心的局部窗口总长度，接近视频首尾时会裁剪 |
| `columns` | `3` | 接触图每行的时间单元数量，控制拼图布局 |

采样基于实际视频帧的 PTS 时间戳，因此实际帧数和时间点不一定等于简单的 FPS 乘法。提高 FPS 会增加图片中的时间单元数量；修改 `columns` 只改变排版，不减少采样数量。

当前有序处理器对全局图和每张局部图均限制为最多 **40 个时间单元**。较长视频如果报采样数量超限，先降低 `coarse_fps`；精修图超限则降低 `refine_fps` 或缩短 `refine_window_ms`。例如 30 秒视频按 2 FPS 采样大约需要 60 个时间单元，应该先降低粗采样率并用准备模式核对实际帧数。API 还限制完整图片 data URI 不超过 10 MiB。

### 4.4 路径与命令行覆盖规则

配置中的 `video`、`output_dir`、三个定义文件路径和 `media_staging_root`，只要是相对路径，都以运行配置文件的目录为基准。例如 `examples/pick_and_place_bolt/ordered_api.yaml` 中的 `../../artifacts/ordered_api` 最终指向仓库的 `artifacts/ordered_api/`。

显式命令行参数会覆盖 YAML 中的对应值。例如临时降低粗采样率，并另存一份结果：

```bash
uv run auto-annotate-ordered \
  --config examples/pick_and_place_bolt/ordered_api.yaml \
  --fps 1 \
  --output-dir artifacts/ordered_api_trial
```

这里命令行传入的 `artifacts/ordered_api_trial` 以**当前工作目录**为基准。YAML 使用 `sampling.coarse_fps`，对应的命令行参数为 `--fps` 或 `--coarse-fps`。YAML 字段名拼错或出现未知字段会被拒绝。

## 5. 先检查接触图，再正式运行

### 第一次运行：准备模式

修改好视频路径和配置后，先运行：

```bash
uv run auto-annotate-ordered \
  --config examples/pick_and_place_bolt/ordered_api.yaml \
  --prepare-only
```

准备模式会读取视频、元数据和三个任务定义文件，生成全局接触图及提示词，并打印所选视角、帧数和文件位置。此时不会调用模型，也不会读取端点或 API key 环境变量。

打开生成的图片，确认视频确实是目标 episode、视角正确、能看到关键动作；再打开提示词，确认步骤顺序、动作定义和边界规则符合要求。准备模式只生成全局图：局部窗口必须等模型给出粗边界以后才能确定，因此此时没有精修图和最终 `output.jsonl`。准备模式成功也不表示模型服务连接已经验证。

### 正式运行：API 或本地服务二选一

API 方式，确认当前终端已经设置环境变量：

```bash
uv run auto-annotate-ordered \
  --config examples/pick_and_place_bolt/ordered_api.yaml
```

本地 vLLM 方式，确认配置中的服务已经启动：

```bash
uv run auto-annotate-ordered \
  --config examples/pick_and_place_bolt/ordered.yaml
```

对于本例的三个动作，正常流程需要一次粗定位请求和两次局部精修请求。程序会保留接触图和提示词，完成全部边界后写出最终结果。单条入口没有批量任务的重试与断点续跑机制；需要这些能力时，使用后面的批量入口。

## 6. 读懂输出，判断分段是否符合预期

### 文件写在哪里

API 示例默认写到仓库的 `artifacts/ordered_api/`。假设输入文件名为 `episode_000000.mp4`，成功运行后的主要文件如下：

```text
artifacts/ordered_api/
├── output.jsonl
├── episode_000000_multiview-contact-sheet.jpg
├── episode_000000_multiview-prompt.txt
├── episode_000000_multiview-refine-transfer-contact-sheet.jpg
├── episode_000000_multiview-refine-transfer-prompt.txt
├── episode_000000_multiview-refine-place-contact-sheet.jpg
└── episode_000000_multiview-refine-place-prompt.txt
```

`output.jsonl` 中这一行对应本次处理的视频。下面命令可以将其格式化显示：

```bash
uv run python -m json.tool artifacts/ordered_api/output.jsonl
```

单条入口固定写入 `<output_dir>/output.jsonl`，不会为下一条视频追加一行；手动处理不同视频时请使用不同输出目录，或者改用批量入口聚合结果。

### 每个字段怎么看

| 字段 | 用途 |
| --- | --- |
| `video_id`、`duration_ms` | 标识本次视频及其时长 |
| `task_summary` | Task 中的整条任务描述 |
| `segments` | 精修后的最终时间段，通常首先查看这里 |
| `coarse_segments` | 精修之前的粗时间段，用于比较 |
| `events`、`coarse_events` | 精修后和精修前的边界事件 |
| `event_refinements` | 每个边界的粗细时间、位移、局部窗口和模型响应 |
| `experiment` | 任务文件哈希、模型信息、采样参数、选用视角等运行记录 |

最终 `segments` 中每段有 `start_ms`、`end_ms`、`values` 和 `sentence` 等字段。例如搬运段可能包含以下内容；这是仅保留关键字段的说明示例：

```json
{
  "start_ms": 3200,
  "end_ms": 7800,
  "values": {
    "atomic_action": "transfer",
    "interaction_object": "workpiece",
    "target": "destination_container"
  },
  "sentence": "搬运：保持对工件的稳定控制并将其移向目标位置的持续过程。"
}
```

检查结果时，将 `transfer` 边界对应到“稳定抓持已建立”的画面，将 `place` 边界对应到“进入最终对位、下放或释放”的画面，再对照相应精修接触图。标签顺序正确本身不代表模型判断准确，因为标签顺序已经由 Task 给定；主要应检查边界时间是否符合 Ontology 规则。真实模型结果可能存在波动，需要正式量化时使用后面的离线评估流程。

## 7. 单条运行遇到问题时

| 现象 | 优先检查 |
| --- | --- |
| 找不到 `schema.yaml`、`ontology.yaml`、`task.yaml` | 从运行配置所在目录解析路径；移动配置时是否同步修改引用 |
| 提示缺少训练掩码、modality 或 episode 元数据 | 输入是否来自完整的 WA2 数据集，是否保留正确的 `videos/chunk-*` 布局 |
| 指定腕部视角后提示该侧未启用 | `training_mask.json` 是否启用该侧动作；未启用时删除对应显式腕部路径 |
| Schema/Ontology ID 不匹配、动作或值校验失败 | Task 引用的 ID、`step_id`、`values.atomic_action` 与词表是否一致 |
| 暂存路径不存在 | 先创建 `backend.media_staging_root` 指向的目录 |
| 接触图超过 40 个时间单元 | 降低粗/细采样率或缩短精修窗口，再用准备模式检查 |
| `refine_fps must be greater ...` | 精修采样率必须大于粗采样率 |
| API 提示缺少环境变量 | 在执行命令的同一终端设置配置所指向的变量名 |
| API 返回 HTTP 错误 | 端点、密钥、工作空间/地域、模型 ID 是否匹配，详见 API 接入文档 |
| vLLM 无法读取图片 | 服务是否可以访问媒体暂存目录及对应本地媒体路径 |
| 视频超过一个 chunk | 当前有序入口仅支持最多 120 秒的单个视频 chunk |
| 边界时间不符合业务预期 | 先检查 Ontology 的边界规则、接触图视角与采样，再看精修前后的位移 |

## 8. 单条确认后，批量运行同一有序任务

批量配置分为 `source`、`processor`、`execution` 和 `output` 四部分。内置 WA2 source 读取 episode 元数据，command processor 为每个 episode 调用 `auto-annotate-ordered`，执行上面的有序动作分段流程。仓库现有 batch 示例通过 `argv` 传入参数；单条运行配置 `ordered.yaml` 与批量配置 `batch.yaml` 是两种不同格式。

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

## 9. 对照人工标注评估边界

准备人工 Gold 集、标注目录和预测 JSONL，修改[评估配置示例](examples/evaluation/wa2_pick_place_boundary_v1.yaml)中的输入、输出路径后运行：

```bash
uv run auto-annotate evaluate \
  --config examples/evaluation/wa2_pick_place_boundary_v1.yaml
```

该示例启用边界误差、时间 IoU 和 episode bootstrap 置信区间，生成机器可读结果、Markdown 报告及复核队列。固定动作顺序任务中，标签和动作序列由任务配置给定，示例关闭这些指标，避免把配置约束当作识别能力。

## 辅助流程：通用标注与 mock 验证

通用入口 `auto-annotate run` 读取 manifest，可用于不固定动作顺序的标注。以下 mock 示例用于验证管线，不调用有序动作处理器。

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
