# -*- coding: utf-8 -*-
import argparse
import sys

# 解析命令行参数
def parse_args():
    '''
    使用示例：
    
    # 1. 基本使用（自动判断数据源）
    python train.py --data_name small
    
    # 2. 明确使用HuggingFace数据集
    python train.py --data_name small --use_huggingface
    python train.py --data_name full --use_huggingface
    python train.py --data_name synthetic_handwrite --use_huggingface
    
    # 3. 使用本地数据集  
    python train.py --data_name small --no_huggingface
    python train.py --data_name local_data_name --no_huggingface
    
    # 4. 从checkpoint恢复训练
    python train.py --data_name small --checkpoint checkpoints/BEST_checkpoint_local_small.pth.tar
    python train.py --data_name small --use_huggingface --checkpoint checkpoints/checkpoint_hf_small.pth.tar
    
    # 5. 使用自定义HuggingFace仓库
    python train.py --data_name small --use_huggingface --hf_repo your_username/your_repo
    
    # 6. 测试数据集加载
    python test_hf_dataset.py
    '''
    parser = argparse.ArgumentParser(description='LaTeX OCR Training Configuration')
    parser.add_argument('--data_name', type=str, default='small', 
                       choices=['small', 'full', 'synthetic_handwrite', 'human_handwrite', 'human_handwrite_print'],
                       help='Dataset name to use (default: small)')
    parser.add_argument('--use_huggingface', action='store_true', default=False,
                       help='Use Hugging Face dataset instead of local data')
    parser.add_argument('--no_huggingface', action='store_true', default=False,
                       help='Use local dataset instead of Hugging Face data')
    parser.add_argument('--hf_repo', type=str, default='linxy/LaTeX_OCR',
                       help='Hugging Face repository name (default: linxy/LaTeX_OCR)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file for resuming training (default: None)')
    
    # 如果在交互式环境中运行或没有命令行参数，使用默认值
    if hasattr(sys, 'ps1') or not sys.argv[1:]:
        return parser.parse_args([])
    return parser.parse_args()

args = parse_args()

#数据路径
data_name = args.data_name  # 数据集名称

# 智能判断是否使用HuggingFace数据集
if args.no_huggingface:
    # 明确指定不使用HuggingFace
    use_huggingface = False
elif args.use_huggingface:
    # 明确指定使用HuggingFace
    use_huggingface = True
else:
    # 默认行为：检查本地数据是否存在，不存在则使用HuggingFace
    import os
    local_vocab_path = f'./data/{data_name}/vocab.json'
    if os.path.exists(local_vocab_path):
        use_huggingface = False
        print(f"[LOCAL] 检测到本地数据集，使用本地数据: ./data/{data_name}/")
    else:
        use_huggingface = True
        print(f"[HF] 未找到本地数据集，将使用HuggingFace数据集: {data_name}")

hf_repo = args.hf_repo  # HuggingFace仓库名称

if use_huggingface:
    # 使用HuggingFace数据集时的配置
    vocab_path = f'./cache/{data_name}/vocab.json'
    train_set_path = f'./cache/{data_name}/train.json'
    val_set_path = f'./cache/{data_name}/val.json'
else:
    # 使用本地数据集时的配置
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
checkpoint = args.checkpoint  # checkpoint文件目录(用于断点继续训练)
save_freq = 2 #保存的间隔
keep_checkpoints = 5  # 保留最近N个checkpoint（不包括best checkpoint）