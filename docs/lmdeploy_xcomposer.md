# LMDeploy 部署 InternLM-XComposer2

InternLM-XComposer2 是一款在自由形式文本图像组合和理解方面表现卓越的视觉语言模型。

InternLM-XComposer2 主要包括四个模型：

| 模型名 | 特点 | 显存（无 KV Cache） | 建议选取的开发机 |
| --- | --- | --- | --- |
| InternLM-XComposer2-4KHD | 4K分辨率，理解，基准，对话 | 15535 MiB | 30% A100（24GB） |
| InternLM-XComposer2-VL-1.8B | 基准、对话 | 3255 MiB | 10% A100（8GB） |
| InternLM-XComposer2 | 图文混合 | 15655 MiB | 30% A100（24GB） |
| InternLM-XComposer2-VL | 基准、对话 | 15535 MiB | 30% A100（24GB） |

> [!IMPORTANT]
> LMDeploy 仅支持了 InternLM-XComposer2 系列模型的视觉对话功能。

## 环境、模型准备

### 配置环境

我们先来配置相关环境。使用如下指令便可以安装好一个 python=3.10 pytorch=2.1.2+cu121 的基础环境了。

```bash
conda create -n lmdeploy python=3.10
conda activate lmdeploy
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

接下来我们安装 LMDeploy。

```bash
pip install lmdeploy[all]==0.4.1
```

### 模型准备

我们以 InternLM-XComposer2-VL 为例，来准备模型权重。

- InternStudio

在 InternStudio，我们可以直接使用如下指令来准备 InternLM-XComposer2-VL 模型权重。

```bash
mkdir -p /root/model
cd /root/model
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b/
```

- 非 InternStudio

在非 InternStudio，我们可以从 ModelScope 下载 InternLM-XComposer2-VL 的权重。

```bash
mkdir -p /root/model
cd /root/model
git lfs install
git clone https://www.modelscope.cn/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b.git
```

## InternLM-XComposer2 推理

### Gradio 在线部署

我们以 InternLM-XComposer2-VL 为例，可以使用如下指令来部署 InternLM-XComposer2-VL 的 Gradio 服务。

```bash
lmdeploy serve gradio /root/model/internlm-xcomposer2-vl-7b/
```

在使用 VSCode 完成端口映射后，我们在本地打开 `http://localhost:6006` 即可看到 InternVL1.5 的 Gradio 服务。

首先通过 Upload Image 上传一张图片，然后在 Instruction 处输入文字，按下回车即可开始对话了。

![img_v3_02aq_f35d7805-888c-4e6b-955b-938db4a3f80g](https://github.com/SmartFlowAI/LLM-Tutorial/assets/75657629/4521bdac-aad8-431a-b4a7-6c16a942f388)

### Pipeline 离线推理

我们也可以使用 `pipeline` 来进行离线推理。新建一个 Python 文件，输入如下代码。然后运行即可。

```python
from lmdeploy.vl import load_image
from lmdeploy import pipeline


pipe = pipeline('/root/model/internlm-xcomposer2-vl-7b/')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmpretrain/main/demo/cat-dog.png')
response = pipe(('请描述图中内容', image))
print(response)
```

![cat-dog](https://raw.githubusercontent.com/open-mmlab/mmpretrain/main/demo/cat-dog.png)

模型输出了

Response(text='在一片花园背景中，一只黑白相间的比熊犬和一只虎斑猫坐在一块色彩斑斓的毯子上。比熊犬和虎斑猫都睁大眼睛，好奇地注视着前方。', generate_token_len=47, input_token_len=1372, session_id=0, finish_reason='stop', token_ids=[60361, 70882, 71266, 68807, 60366, 60353, 70334, 73784, 60544, 71991, 60505, 62234, 62645, 60381, 70334, 61897, 62035, 61519, 70689, 70516, 69959, 62035, 65430, 60354, 63521, 71024, 60355, 60505, 62234, 62645, 60381, 61897, 62035, 61519, 60406, 63525, 60368, 68948, 60353, 73718, 60415, 60551, 60690, 60486, 75300, 60355], logprobs=None)

可以看到，模型对图片的描述虽然不如 InternVL1.5 那样细节，但也是非常准确的。
