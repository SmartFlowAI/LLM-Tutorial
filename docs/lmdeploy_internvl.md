# LMDeploy 部署 InternVL1.5

InternVL1.5 是 OpenGVLab 最新开源的视觉多模态大模型，从评测角度看，InternVL1.5 是目前最好的开源视觉多模态大模型。InternVL 包括一个 6B 大的视觉模型 InternViT 和一个 20B 大的语言模型 InternLM2-Chat-20B。

LMDeploy 团队也已经支持了 InternVL1.5 的部署，下面是部署的详细步骤。

部署时所需显存 > 40GB（无 KV Cache 时 47695 MiB），请使用 100% A100，即 80GB 显存 A100。

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
conda activate lmdeploy
pip install lmdeploy[all]==0.4.1
```

### 模型准备

在开始部署前，我们首先来准备 InternVL1.5 模型权重。

- InternStudio

在 InternStudio，我们可以直接使用如下指令来准备 InternVL1.5 模型权重。

```bash
mkdir -p /root/model
cd /root/model
ln -s /root/share/new_models/OpenGVLab/InternVL-Chat-V1-5 .
```

- 非 InternStudio

在非 InternStudio，我们可以从 HuggingFace 上下载 InternVL1.5 的权重。

```bash
mkdir -p /root/model
cd /root/model
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5
```

## InternVL1.5 推理

### Gradio 在线部署

我们可以使用如下指令来部署 InternVL1.5 的 Gradio 服务。

```bash
lmdeploy serve gradio /root/model/InternVL-Chat-V1-5
```

在使用 VSCode 完成端口映射后，我们在本地打开 `http://localhost:6006` 即可看到 InternVL1.5 的 Gradio 服务。

首先通过 Upload Image 上传一张图片，然后在 Instruction 处输入文字，按下回车即可开始对话了。

![gU5tZkNyBw](https://github.com/SmartFlowAI/LLM-Tutorial/assets/75657629/2747908e-2c0e-4191-ba46-b540f0c73c77)

![img_v3_02ap_f33eaf64-dc12-4f48-8192-a7479ea0930g](https://github.com/SmartFlowAI/LLM-Tutorial/assets/75657629/59c8e712-9d59-4d3d-b26a-ad509682f29d)

两张图分别展示了 InternVL1.5 模型遵循图中指令回复的能力以及其知识能力。

### Pipeline 离线推理

我们也可以使用 `pipeline` 来进行离线推理。新建一个 Python 文件，输入如下代码。然后运行即可。

```python
from lmdeploy.vl import load_image
from lmdeploy import pipeline


pipe = pipeline('/root/model/InternVL-Chat-V1-5')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmpretrain/main/demo/cat-dog.png')
response = pipe(('请描述图中内容', image))
print(response)
```

![cat-dog](https://raw.githubusercontent.com/open-mmlab/mmpretrain/main/demo/cat-dog.png)

模型输出了

Response(text='这张图片展示了一只小猎犬和一只小猫坐在一起，它们看起来都很友好和好奇。猎犬有着黑色、棕色和白色的毛皮，而小猫则是灰色的。它们坐在一条彩色的条纹布上，背景是户外的绿色草地和一些花朵。两只动物都直视着镜头，表情显得有些无辜和可爱。整体上，这张照片给人一种温馨和谐的感觉。', generate_token_len=80, input_token_len=1831, session_id=0, finish_reason='stop', token_ids=[72998, 68467, 69552, 86869, 60398, 62593, 62645, 60381, 70334, 81890, 61213, 68769, 60353, 69290, 69835, 69907, 75184, 60381, 73718, 60355, 62593, 62645, 69703, 69329, 60359, 75794, 60381, 74100, 61101, 60834, 60353, 60458, 81890, 71489, 87542, 60355, 69290, 70689, 69395, 61263, 68738, 79063, 60761, 60370, 60353, 68807, 60357, 60746, 69881, 69391, 80161, 81701, 76039, 60355, 75021, 69121, 60406, 60578, 60690, 60486, 71499, 60353, 70407, 70213, 68568, 85815, 60381, 69730, 60355, 69217, 60370, 60353, 72998, 68805, 80255, 71869, 71232, 69440, 60355], logprobs=None)

可以看到，模型对图片的描述是非常准确的。