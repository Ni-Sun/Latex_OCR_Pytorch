# Kaggle训练指南 (Kaggle Training Guide)

本指南说明如何在Kaggle上进行增量训练，以应对12小时运行时间限制。

## 新增功能 (New Features)

### 1. 自动打包Checkpoints
每次保存checkpoint后，系统会自动创建两个压缩包：
- `checkpoints_{timestamp}.tar.gz` - 带时间戳的备份
- `checkpoints.tar.gz` - 最新版本（固定名称，方便下载）

这样即使训练中断，也能获得最新的checkpoint文件。

### 2. 训练日志保存
所有训练输出会同时保存到 `training.log` 文件中，包括：
- 每个epoch的训练和验证指标
- checkpoint保存记录
- 学习率调整信息
- 错误和警告信息

即使Kaggle因为cell未完成而不显示输出，日志文件仍会保留所有信息。

### 3. 增量训练支持
新增 `--max_epochs` 参数，可以指定单次运行训练的epoch数量：

```bash
# 首次训练5个epoch
python train.py --data_name human_handwrite --use_huggingface --max_epochs 5

# 从checkpoint继续训练5个epoch
python train.py --data_name human_handwrite --use_huggingface --max_epochs 5 \
    --checkpoint checkpoints/BEST_checkpoint_hf_human_handwrite.pth.tar
```

## Kaggle使用示例

### Cell 1: 环境准备
```bash
%%bash
pip install editdistance

# 复制项目到working目录
if [ ! -d "/kaggle/working/latex-ocr-pytorch" ]; then
    cp -r /kaggle/input/latex-ocr-pytorch /kaggle/working
fi
```

### Cell 2: 训练（可多次运行）
```bash
%%bash
cd /kaggle/working/latex-ocr-pytorch

# 第一次运行：不指定checkpoint
python train.py --data_name human_handwrite --use_huggingface --max_epochs 5

# 后续运行：从checkpoint继续
# python train.py --data_name human_handwrite --use_huggingface --max_epochs 5 \
#     --checkpoint checkpoints/BEST_checkpoint_hf_human_handwrite.pth.tar
```

### Cell 3: 查看结果
```bash
%%bash
cd /kaggle/working/latex-ocr-pytorch

# 查看压缩包
ls -lh checkpoints*.tar.gz

# 查看训练日志
tail -50 training.log
```

## 训练策略建议

1. **首次训练**: 使用 `--max_epochs 5` 训练5个epoch
2. **检查结果**: 查看 `training.log` 确认训练正常
3. **继续训练**: 如果时间允许，在新cell中使用相同命令继续训练
4. **下载checkpoint**: 时间快到12小时时，下载 `checkpoints.tar.gz`
5. **下次会话**: 上传checkpoint作为dataset，继续训练

## 重要提示

- checkpoint文件会自动管理，只保留最近5个 + 1个最佳checkpoint
- 每次保存checkpoint都会自动打包，无需手动操作
- 日志文件会持续追加，记录所有训练会话
- 使用 `--checkpoint` 参数时，确保checkpoint文件存在

## 参数说明

- `--data_name`: 数据集名称（small, full, synthetic_handwrite, human_handwrite等）
- `--use_huggingface`: 使用HuggingFace数据集
- `--max_epochs`: 本次运行最多训练多少个epoch（新增）
- `--checkpoint`: checkpoint文件路径（用于继续训练）
- `--hf_repo`: HuggingFace仓库名称（默认：linxy/LaTeX_OCR）

## 故障排除

如果遇到问题：

1. **查看日志**: `cat training.log` 查看完整日志
2. **检查checkpoint**: `ls -lh checkpoints/` 查看checkpoint文件
3. **解压测试**: `tar -tzf checkpoints.tar.gz` 查看压缩包内容
4. **空间不足**: 清理旧的时间戳压缩包 `rm checkpoints_*.tar.gz`
