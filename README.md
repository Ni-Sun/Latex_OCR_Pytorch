# Latex_OCR_Pytorch

支持从HuggingFace直接加载数据集的LaTeX OCR模型！

主要是这个版本的Pytorch实现:
[LinXueyuanStdio/LaTeX_OCR_PRO](https://github.com/LinXueyuanStdio/LaTeX_OCR_PRO)

## HuggingFace数据集支持

现在可以直接从HuggingFace加载数据集，无需手动下载和预处理。默认使用 [linxy/LaTeX_OCR](https://huggingface.co/datasets/linxy/LaTeX_OCR) 的数据集

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

## ⚙️ 命令行参数详解

### 🔧 快速参考

```bash
# 基本用法（默认数据集 small，自动判断数据源）
python train.py

# 指定数据集
python train.py --data_name full

# 明确使用 HuggingFace 数据集
python train.py --data_name small --use_huggingface

# 强制使用本地数据集
python train.py --data_name small --no_huggingface

# 从 checkpoint 恢复训练
python train.py --data_name small --checkpoint checkpoints/BEST_checkpoint_local_small.pth.tar

# 使用自定义 HuggingFace 仓库
python train.py --data_name small --hf_repo your_username/your_repo
```

### 参数详解

| 参数                  | 类型   | 默认值              | 说明                                                                                                          |
| --------------------- | ------ | ------------------- | ------------------------------------------------------------------------------------------------------------- |
| `--data_name`       | 字符串 | `small`           | 数据集名称，支持:`small`, `full`, `synthetic_handwrite`, `human_handwrite`, `human_handwrite_print` |
| `--use_huggingface` | 标志   | -                   | 强制使用 HuggingFace 数据集                                                                                   |
| `--no_huggingface`  | 标志   | -                   | 强制使用本地数据集                                                                                            |
| `--hf_repo`         | 字符串 | `linxy/LaTeX_OCR` | HuggingFace 仓库名                                                                                            |
| `--checkpoint`      | 字符串 | `None`            | checkpoint 文件路径，用于恢复训练                                                                             |
| `--max_epochs`      | 整数   | `None`            | 单次运行的最大 epoch 数（可选，用于增量训练）                                                                 |

### 📊 可用的数据集

来自 `linxy/LaTeX_OCR` HuggingFace 仓库：

| 数据集名称                | 样本数 | 描述                                                        |
| ------------------------- | ------ | ----------------------------------------------------------- |
| `small`                 | ~110   | 小型数据集，样本数较少，适合快速测试                        |
| `full`                  | ~100k  | 完整印刷体数据集，基于 LaTeX 渲染的约 100k 样本             |
| `synthetic_handwrite`   | ~100k  | 合成手写数据，基于 full 的公式用手写字体合成                |
| `human_handwrite`       | 较小   | 真实手写数据，主要来源于 CROHME，更符合电子屏手写体         |
| `human_handwrite_print` | 较小   | 手写印刷混合数据，公式同 human_handwrite，图片由 LaTeX 渲染 |

### 🎯 数据源选择策略

#### 1. 默认智能模式（推荐）

```bash
python train.py --data_name small
```

**工作流程：**

1. 检查本地是否存在 `./data/small/vocab.json`
2. 如果存在 → 使用本地数据集
3. 如果不存在 → 自动使用 HuggingFace 数据集

#### 2. 明确使用 HuggingFace 数据集

```bash
python train.py --data_name small --use_huggingface
```

- 强制从 HuggingFace 下载和使用数据集
- 首次使用会自动下载并缓存到 `./cache/{data_name}/`
- 后续使用直接从缓存加载

#### 3. 强制使用本地数据集

```bash
python train.py --data_name small --no_huggingface
```

- 只使用 `./data/{data_name}/` 中的本地数据
- 如果本地数据不存在会报错
- 适合使用自定义或私有数据集

### 📁 数据缓存位置

**HuggingFace 模式下的缓存结构：**

```
./cache/
├── small/
│   ├── vocab.json          # 词汇表
│   ├── train.json          # 训练数据
│   ├── val.json            # 验证数据
│   └── images/             # 图片文件
├── full/
│   ├── vocab.json
│   ├── train.json
│   ├── val.json
│   └── images/
└── ... 其他数据集 ...
```

**本地模式下的文件结构：**

```
./data/
├── small/
│   ├── vocab.json
│   ├── train.json          # 如果有分离的文件
│   ├── val.json
│   └── images/
├── full/
│   ├── vocab.json
│   ├── data.json           # 或统一的数据文件
│   └── images/
└── ... 其他数据集 ...
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
3. HuggingFace datasets >= 2.0

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

## To do

- [ ] 推断部分
- [ ] Attention层的可视化
- [X] 预训练模型
- [X] 打包的训练数据
- [ ] perplexity指标
```
