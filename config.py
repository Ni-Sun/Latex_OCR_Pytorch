import argparse
import sys

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='LaTeX OCR Training Configuration')
    parser.add_argument('--data_name', type=str, default='small', 
                       choices=['small', 'CROHME', 'full', 'hand', 'fullhand'],
                       help='Dataset name to use (default: small)')
    
    # 如果在交互式环境中运行或没有命令行参数，使用默认值
    if hasattr(sys, 'ps1') or not sys.argv[1:]:
        return parser.parse_args([])
    return parser.parse_args()

args = parse_args()

#数据路径
data_name = args.data_name  # 模型名称,仅在保存的时候用到
vocab_path = f'./data/{data_name}/vocab.json'

# 检查是否存在分离的train.json和val.json文件，否则使用data.json
import os
train_json_path = f'./data/{data_name}/train.json'
val_json_path = f'./data/{data_name}/val.json'
data_json_path = f'./data/{data_name}/data.json'

if os.path.exists(train_json_path) and os.path.exists(val_json_path):
    train_set_path = train_json_path
    val_set_path = val_json_path
else:
    # 如果没有分离的文件，使用data.json（需要在dataloader中处理）
    train_set_path = data_json_path
    val_set_path = data_json_path


# 模型参数
emb_dim = 30  # 词嵌入维数80
attention_dim = 128  # attention 层维度 256
decoder_dim = 128  # decoder维度 128
dropout = 0.5
buckets = [[240, 100], [320, 80], [400, 80], [400, 100], [480, 80], [480, 100],
        [560, 80], [560, 100], [640, 80], [640, 100], [720, 80], [720, 100],
        [720, 120], [720, 200], [800, 100], [800, 320], [1000, 200],
        [1000, 400], [1200, 200], [1600, 200],
        ]


# 训练参数
start_epoch = 0
epochs = 250  # 不触发早停机制时候最大迭代次数
epochs_since_improvement = 0  # 用于跟踪在验证集上分数没有提高的迭代次数
batch_size = 1 #训练解批大小
test_batch_size = 2 #验证集批大小
encoder_lr = 1e-4  # 学习率
decoder_lr = 4e-4  # 学习率
grad_clip = 5.  # 梯度裁剪阈值
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_score = 0.  # 目前最好的 score 
print_freq = 100  # 状态的批次打印间隔
# checkpoint = f'BEST_checkpoint_{data_name}.pth.tar'  # checkpoint文件目录(用于断点继续训练)
checkpoint = None  # checkpoint文件目录(用于断点继续训练)
save_freq = 2 #保存的间隔