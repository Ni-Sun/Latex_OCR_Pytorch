Latex_OCR_Pytorch

主要是这个版本的Pytorch实现:

[LinXueyuanStdio/LaTeX_OCR_PRO](https://github.com/LinXueyuanStdio/LaTeX_OCR_PRO)

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

## GitHub Actions 自动化训练

本项目提供了 GitHub Actions 工作流，可以在云端自动化训练模型。

### 使用方法

1. 进入 GitHub 仓库的 **Actions** 标签页
2. 选择 **"手动训练工作流"**
3. 点击 **"Run workflow"** 按钮
4. 在下拉菜单中选择数据集类型：
   - `small`: 小型数据集（默认）
   - `CROHME`: CROHME 数据集
   - `full`: 完整数据集
   - `hand`: 手写数据集
   - `fullhand`: 完整手写数据集
5. 点击绿色的 **"Run workflow"** 按钮开始训练

### 功能特性

- ✅ **手动触发**: 通过 GitHub Actions 界面手动启动训练
- ✅ **数据集选择**: 支持 5 种不同的数据集类型
- ✅ **自动上传模型**: 训练完成后自动上传模型检查点作为 artifacts
- ✅ **超时保护**: 设置了 6 小时的超时限制，防止无限运行
- ✅ **依赖管理**: 自动安装所需的 Python 依赖（PyTorch, OpenCV, NumPy 等）
- ✅ **数据验证**: 训练前自动检查数据集文件是否存在

### 下载训练好的模型

训练完成后，可以在 Actions 运行页面的 **Artifacts** 部分下载模型检查点：
- `checkpoint_{数据集名称}.pth.tar`: 最新的检查点
- `BEST_checkpoint_{数据集名称}.pth.tar`: 最佳模型检查点

Artifacts 保留 30 天。

### 注意事项

⚠️ **重要**: 使用 GitHub Actions 训练前，请确保数据集文件已经存在于仓库的 `data/{数据集名称}/` 目录中。需要的文件包括：
- `vocab.json`: 词汇表文件
- `train.json` 和 `val.json`: 训练和验证数据（或）
- `data.json`: 统一的数据文件（对于某些数据集）

## To do

- [ ] 推断部分
- [ ] Attention层的可视化
- [X] 预训练模型
- [X] 打包的训练数据
- [ ] perplexity指标
