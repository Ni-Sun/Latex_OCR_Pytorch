# Latex_OCR_Pytorch

支持从HuggingFace直接加载数据集的LaTeX OCR模型！

主要是这个版本的Pytorch实现:
[LinXueyuanStdio/LaTeX_OCR_PRO](https://github.com/LinXueyuanStdio/LaTeX_OCR_PRO)

## 🆕 新功能：HuggingFace数据集支持

现在可以直接从HuggingFace加载数据集，无需手动下载和预处理：

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 使用HuggingFace数据集训练 (推荐)
python train.py --data_name small --use_huggingface

# 使用其他数据集
python train.py --data_name full --use_huggingface
python train.py --data_name synthetic_handwrite --use_huggingface
python train.py --data_name human_handwrite --use_huggingface
python train.py --data_name human_handwrite_print --use_huggingface
```

### 📊 可用的HuggingFace数据集

来自 `linxy/LaTeX_OCR` 仓库：

- `small`: 小型数据集，适合快速测试
- `full`: 完整数据集，较大规模
- `synthetic_handwrite`: 合成手写数据
- `human_handwrite`: 真实手写数据
- `human_handwrite_print`: 手写印刷混合数据

### ⚙️ 命令行参数

```bash
python train.py [选项]

选项:
  --data_name {small,full,synthetic_handwrite,human_handwrite,human_handwrite_print}
                        数据集名称 (默认: small)
  --use_huggingface     使用HuggingFace数据集 (默认: True)  
  --hf_repo             HuggingFace仓库名 (默认: linxy/LaTeX_OCR)
  --checkpoint          checkpoint文件路径用于继续训练
```

### 📁 数据缓存

首次使用时，数据会被下载并缓存到本地：

```
./cache/
├── small/
│   ├── vocab.json
│   ├── train.json
│   ├── val.json
│   └── images/
├── full/
└── ...
```

### 🧪 测试功能

```bash
# 测试数据集加载
python test_hf_dataset.py

# 查看使用示例
python usage_example.py
```

## 📖 命令行参数详细说明

### 🎯 概述

LaTeX OCR 项目现在支持灵活的数据源配置，既可以使用本地数据集，也可以从 HuggingFace 自动加载数据集。

### 🔧 命令行参数详解

#### 基本参数

| 参数                  | 类型   | 默认值              | 说明                    |
| --------------------- | ------ | ------------------- | ----------------------- |
| `--data_name`       | 选择   | `small`           | 数据集名称              |
| `--use_huggingface` | 标志   | -                   | 使用 HuggingFace 数据集 |
| `--no_huggingface`  | 标志   | -                   | 使用本地数据集          |
| `--hf_repo`         | 字符串 | `linxy/LaTeX_OCR` | HuggingFace 仓库名      |
| `--checkpoint`      | 字符串 | `None`            | checkpoint 文件路径     |

#### 可用数据集

| 数据集名称                | 描述             | 推荐用途      |
| ------------------------- | ---------------- | ------------- |
| `small`                 | 小型数据集       | 🧪 快速测试   |
| `full`                  | 完整印刷体数据集 | 📚 正式训练   |
| `synthetic_handwrite`   | 合成手写数据     | ✍️ 手写识别 |
| `human_handwrite`       | 真实手写数据     | 👤 人工手写   |
| `human_handwrite_print` | 手写印刷混合     | 🔄 混合场景   |
| `CROHME`                | CROHME竞赛数据   | 🏆 基准测试   |
| `hand`                  | 本地手写数据     | 💾 本地数据   |
| `fullhand`              | 完整手写数据     | 💾 本地数据   |

### 🚀 使用场景详解

#### 1️⃣ 默认智能模式（推荐）

```bash
python train.py --data_name small
```

**工作流程：**

1. 🔍 检查本地是否存在 `./data/small/vocab.json`
2. ✅ 如果存在 → 使用本地数据集
3. ❌ 如果不存在 → 自动使用 HuggingFace 数据集

**优点：** 自动选择最合适的数据源，无需手动判断

#### 2️⃣ 明确使用 HuggingFace 数据集

```bash
python train.py --data_name small --use_huggingface
python train.py --data_name synthetic_handwrite --use_huggingface
```

**特点：**

- 🌐 强制从 HuggingFace 下载和使用数据集
- 📦 首次使用会自动下载并缓存到 `./cache/{data_name}/`
- 🔄 后续使用直接从缓存加载

#### 3️⃣ 强制使用本地数据集

```bash
python train.py --data_name small --no_huggingface
python train.py --data_name CROHME --no_huggingface
```

**特点：**

- 💾 只使用 `./data/{data_name}/` 中的本地数据
- ⚠️ 如果本地数据不存在会报错
- 🔒 适合使用自定义或私有数据集

#### 4️⃣ 从 checkpoint 恢复训练

```bash
python train.py --data_name small --checkpoint checkpoints/BEST_checkpoint_local_small.pth.tar
python train.py --data_name small --use_huggingface --checkpoint checkpoints/checkpoint_hf_small.pth.tar
```

#### 5️⃣ 使用自定义 HuggingFace 仓库

```bash
python train.py --data_name custom_data --use_huggingface --hf_repo your_username/your_latex_ocr_repo
```

### 📁 文件结构说明

#### HuggingFace 数据集模式

```
./cache/
├── small/
│   ├── vocab.json          # 词汇表
│   ├── train.json          # 训练数据
│   ├── val.json            # 验证数据
│   └── images/             # 图片缓存
│       ├── train_000001.png
│       ├── val_000001.png
│       └── ...
└── synthetic_handwrite/
    └── ...
```

#### 本地数据集模式

```
./data/
├── small/
│   ├── vocab.json          # 词汇表
│   ├── train.json          # 训练数据 (如果分离)
│   ├── val.json            # 验证数据 (如果分离)
│   └── data.json           # 统一数据文件 (如果未分离)
└── CROHME/
    └── ...
```

### ⚡ 最佳实践

#### 🧪 开发和测试

```bash
# 快速测试（小数据集）
python train.py --data_name small

# 功能验证
python test_config.py --data_name small --use_huggingface
```

#### 🏋️ 正式训练

```bash
# 完整数据集训练
python train.py --data_name full --use_huggingface

# 手写数据训练
python train.py --data_name human_handwrite --use_huggingface
```

#### 🔄 继续训练

```bash
# 从最佳checkpoint继续
python train.py --data_name full --use_huggingface --checkpoint checkpoints/BEST_checkpoint_full.pth.tar

# 从普通checkpoint继续
python train.py --data_name full --use_huggingface --checkpoint checkpoints/checkpoint_full.pth.tar
```

#### 🛡️ 离线环境

```bash
# 强制使用本地数据（不依赖网络）
python train.py --data_name small --no_huggingface
```

### 🔍 诊断工具

#### 检查配置

```bash
python test_config.py                              # 默认配置
python test_config.py --use_huggingface           # HuggingFace模式
python test_config.py --no_huggingface            # 本地模式
python test_config.py --data_name synthetic_handwrite  # 自动判断
```

#### 测试数据加载

```bash
python test_hf_dataset.py                         # 测试HuggingFace加载
python usage_example.py                           # 查看使用示例
```

#### 查看帮助

```bash
python train.py --help                            # 查看所有参数
```

## 原有功能

感谢@LinXueyuanStdio 的工作以及指导.本项目与上述项目思路一致，但在实现上修改了一些地方:

* 数据集的重新定义,但使用原有类似的预处理方式
* 代码简化，目前仅保留主要部分，命令行控制等在后续补充
* 内存优化，相对较少的内存需求，支持较大批量的训练。但批大小一样的情况下实测速度提高不大
* 使用Checkpoint特性，在编码过程中出现OOM则自动进行分段计算
* 在训练时候采用贪婪策略，Beam Search仅在推断时候采用
* Scheduled Sampling策略

Follow these paper:

1. [Show, Attend and Tell(Kelvin Xu...)](https://arxiv.org/abs/1502.03044)
2. [Harvard&#39;s paper and dataset](http://lstm.seas.harvard.edu/latex/)

Follow these tutorial:

1. [Seq2Seq for LaTeX generation](https://guillaumegenthial.github.io/image-to-latex.html).
2. [a PyTorch Tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).

## 环境

1. Python >= 3.6
2. Pytorch >= 1.2
3. HuggingFace datasets >= 2.0 (新增)

## 数据

使用[LinXueyuanStdio/Data-for-LaTeX_OCR](https://github.com/LinXueyuanStdio/Data-for-LaTeX_OCR) 数据集,原仓库较大,后续提供打包下载.

已包括上述仓库中small数据集
印刷体数据全集[百度云](https://pan.baidu.com/s/1xIsgHDhVu85L8cGdqqG7kw) 提取码：tapj [Google Drive](https://drive.google.com/open?id=1THp_O7uwavcjsnQXsxx_JPvYn9-gml7T)
自己划分的混合CROHME2011,2012数据集[Google Drive](https://drive.google.com/open?id=1KgpAzA7k8ayjPTstin6M8ykGsW8GR9bu)

## 训练模型

在自己划分CROHME2011,2012数据集上使用以下参数的训练模型[Google Drive](https://drive.google.com/open?id=1_geqm9a86TJKK9RpZ39d9X5655s4NXa9)
emb_dim = 30
attention_dim = 128
decoder_dim = 128
后续补充模型测试结果以及colab

## 数据格式

数据集文件生成参考[utils.py](./model/utils.py)的get_latex_ocrdata

数据集文件json格式,包括训练集文件,验证集文件,字典文件.

字典格式:

python字典(符号——编号)的json储存

数据集格式:

```
```shell
训练/验证数据集
├── file_name1 图片文件名 str
│   ├── img_path:文件路径(到文件名,含后缀) str
│   ├── size:图片尺寸 [长,宽] list
│   ├── caption:图片代表的公式,各个符号之间必须要空格分隔 str
│   └── caption_len:len(caption.split()) int
|   ...
eg:
{
"0.png":
    {
    "img_path":"./mydata/0.png",
    "size":[442,62],
    "caption":"\frac { a + b } { 2 }",
    "caption_len":9,
    }
"2.png":...
}

```

```

图片预处理

参考dataloader/data_turn主要进行以下操作

1. 灰度化
2. 裁剪公式部分
3. 上下左右各padding 8个像素
4. `[可选]`下采样

## 使用说明

### 数据集选择

现在可以通过命令行参数来选择不同的数据集进行训练：

#### 默认使用 small 数据集
```bash
python train.py
```

#### 使用其他数据集

```bash
# 使用 CROHME 数据集
python train.py --data_name CROHME

# 使用 full 数据集
python train.py --data_name full

# 使用 hand 数据集
python train.py --data_name hand

# 使用 fullhand 数据集
python train.py --data_name fullhand
```

#### 可用的数据集选项

- `small`: 小型数据集（默认）
- `CROHME`: CROHME 数据集
- `full`: 完整数据集
- `hand`: 手写数据集
- `fullhand`: 完整手写数据集

#### 查看帮助

```bash
python train.py --help
```

### 配置文件说明

修改后的配置文件会根据 `--data_name` 参数自动设置以下路径：

- `vocab_path`: `./data/{data_name}/vocab.json`
- `train_set_path` 和 `val_set_path`:
  - 如果存在 `train.json` 和 `val.json` 文件（如 `small` 数据集），则分别使用这些文件
  - 如果不存在分离的文件（如 `full`, `hand`, `fullhand` 数据集），则使用 `data.json` 文件

保存的模型文件名也会包含数据集名称：

- 格式：`checkpoint_{data_name}.pth.tar`
- 最佳模型：`BEST_checkpoint_{data_name}.pth.tar`

### 数据集文件结构

不同数据集的文件结构：

#### small 数据集（分离文件）

```
data/small/
├── vocab.json
├── train.json
├── val.json
└── test.json
```

#### full, hand, fullhand 数据集（统一文件）

```
data/full/
├── vocab.json
├── data.json
└── ...
```

## To do

- [ ] 推断部分
- [ ] Attention层的可视化
- [X] 预训练模型
- [X] 打包的训练数据
- [ ] perplexity指标
