import os
import numpy as np
import json
import cv2
import torch
from datasets import load_dataset
import tempfile
import urllib.request
from PIL import Image
import logging
import sys

def setup_logger(log_file='training.log'):
    """
    设置日志记录器，同时输出到控制台和文件
    用于Kaggle环境保存训练日志
    
    :param log_file: 日志文件路径
    :return: logger对象
    """
    # 创建logger
    logger = logging.getLogger('latex_ocr_training')
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建文件handler
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # 创建控制台handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加handler到logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def load_json(path):
    with open(path,'r')as f:
        data = json.load(f)
    return data

def cal_word_freq(vocab,formuladataset):
    #统计词频用于计算perplexity
    word_count = {}
    for i in vocab.values():
        word_count[i] = 0
    count = 0
    for i in formuladataset.data.values():
        words = i['caption'].split()
        for j in words:
            word_count[vocab[j]] += 1
            count += 1
    for i in word_count.keys():
        word_count[i] = word_count[i]/count
    return word_count

def load_huggingface_dataset(hf_repo, data_name, cache_dir='./cache'):
    """
    从HuggingFace加载数据集并转换为本地格式
    
    Args:
        hf_repo: HuggingFace仓库名称 (如 'linxy/LaTeX_OCR')
        data_name: 数据集名称 (如 'small', 'full', 'synthetic_handwrite'等)
        cache_dir: 缓存目录
    
    Returns:
        vocab: 词汇表字典
        train_data: 训练数据字典
        val_data: 验证数据字典
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("请安装datasets库: pip install datasets")
        return None, None, None
    
    # 创建缓存目录
    cache_path = os.path.join(cache_dir, data_name)
    os.makedirs(cache_path, exist_ok=True)
    
    print(f"正在从HuggingFace加载数据集: {hf_repo}/{data_name}")
    
    try:
        # 加载数据集
        dataset = load_dataset(hf_repo, data_name)
        
        vocab_temp = set()
        train_data = {}
        val_data = {}
        
        # 处理训练集
        if 'train' in dataset:
            print(f"处理训练集: {len(dataset['train'])} 个样本")
            for idx, item in enumerate(dataset['train']):
                img_filename = f"train_{idx:06d}.png"
                img_path = os.path.join(cache_path, 'images', img_filename)
                
                # 创建图片目录
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                
                # 保存图片
                if 'image' in item:
                    try:
                        if hasattr(item['image'], 'save'):  # PIL Image
                            item['image'].save(img_path)
                        else:  # 可能是numpy数组或其他格式
                            img = Image.fromarray(item['image'])
                            img.save(img_path)
                        
                        # 获取图片尺寸
                        img = cv2.imread(img_path)
                        if img is not None:
                            size = (img.shape[1], img.shape[0])
                        else:
                            continue
                            
                    except Exception as e:
                        print(f"保存图片失败 {img_filename}: {e}")
                        continue
                else:
                    continue
                
                # 处理标签
                if 'latex' in item:
                    caption = item['latex'].strip()
                elif 'text' in item:
                    caption = item['text'].strip()
                else:
                    continue
                
                # 添加到词汇表
                for token in caption.split():
                    vocab_temp.add(token)
                
                train_data[img_filename] = {
                    'img_path': img_path,
                    'size': size,
                    'caption': caption,
                    'caption_len': len(caption.split()) + 2  # 加上<start>和<end>
                }
        
        # 处理验证集
        if 'validation' in dataset:
            print(f"处理验证集: {len(dataset['validation'])} 个样本")
            for idx, item in enumerate(dataset['validation']):
                img_filename = f"val_{idx:06d}.png"
                img_path = os.path.join(cache_path, 'images', img_filename)
                
                # 保存图片
                if 'image' in item:
                    try:
                        if hasattr(item['image'], 'save'):  # PIL Image
                            item['image'].save(img_path)
                        else:  # 可能是numpy数组或其他格式
                            img = Image.fromarray(item['image'])
                            img.save(img_path)
                        
                        # 获取图片尺寸
                        img = cv2.imread(img_path)
                        if img is not None:
                            size = (img.shape[1], img.shape[0])
                        else:
                            continue
                            
                    except Exception as e:
                        print(f"保存图片失败 {img_filename}: {e}")
                        continue
                else:
                    continue
                
                # 处理标签
                if 'latex' in item:
                    caption = item['latex'].strip()
                elif 'text' in item:
                    caption = item['text'].strip()
                else:
                    continue
                
                # 添加到词汇表
                for token in caption.split():
                    vocab_temp.add(token)
                
                val_data[img_filename] = {
                    'img_path': img_path,
                    'size': size,
                    'caption': caption,
                    'caption_len': len(caption.split()) + 2
                }
        elif 'test' in dataset:
            # 如果没有validation但有test，用test作为验证集
            print(f"处理测试集作为验证集: {len(dataset['test'])} 个样本")
            for idx, item in enumerate(dataset['test']):
                img_filename = f"val_{idx:06d}.png"
                img_path = os.path.join(cache_path, 'images', img_filename)
                
                # 保存图片
                if 'image' in item:
                    try:
                        if hasattr(item['image'], 'save'):  # PIL Image
                            item['image'].save(img_path)
                        else:
                            img = Image.fromarray(item['image'])
                            img.save(img_path)
                        
                        # 获取图片尺寸
                        img = cv2.imread(img_path)
                        if img is not None:
                            size = (img.shape[1], img.shape[0])
                        else:
                            continue
                            
                    except Exception as e:
                        print(f"保存图片失败 {img_filename}: {e}")
                        continue
                else:
                    continue
                
                # 处理标签
                if 'latex' in item:
                    caption = item['latex'].strip()
                elif 'text' in item:
                    caption = item['text'].strip()
                else:
                    continue
                
                # 添加到词汇表
                for token in caption.split():
                    vocab_temp.add(token)
                
                val_data[img_filename] = {
                    'img_path': img_path,
                    'size': size,
                    'caption': caption,
                    'caption_len': len(caption.split()) + 2
                }
        
        # 生成词汇表
        vocab = {}
        vocab['<pad>'] = 0
        vocab['<start>'] = 1
        vocab['<end>'] = 2
        vocab['<unk>'] = 3
        
        for i, token in enumerate(sorted(vocab_temp)):
            vocab[token] = i + 4
        
        # 保存词汇表和数据
        vocab_path = os.path.join(cache_path, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
        
        train_path = os.path.join(cache_path, 'train.json')
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        val_path = os.path.join(cache_path, 'val.json')
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        print(f"数据集处理完成:")
        print(f"  - 训练集: {len(train_data)} 个样本")
        print(f"  - 验证集: {len(val_data)} 个样本")
        print(f"  - 词汇表大小: {len(vocab)}")
        print(f"  - 缓存路径: {cache_path}")
        
        return vocab, train_data, val_data
        
    except Exception as e:
        print(f"加载HuggingFace数据集失败: {e}")
        return None, None, None


def get_latex_ocrdata(path, mode='val'):
    """
    加载LaTeX OCR数据并生成词汇表
    
    Args:
        path: 数据路径
        mode: 模式 ('val', 'train', 'test')
    
    Returns:
        vocab: 词汇表字典
        data: 数据字典
    """
    assert mode in ['val','train','test']
    match = []
    with open(path + 'matching/'+mode+'.matching.txt','r')as f:
        for i in f.readlines():
            match.append(i[:-1])

    formula = []
    with open(path + 'formulas/'+mode+'.formulas.norm.txt','r')as f:
        for i in f.readlines():
            formula.append(i[:-1])

    vocab_temp = set()
    data = {}

    for i in match:
        img_path = path + 'images/images_' + mode + '/' + i.split()[0]
        try:
            img = cv2.imread(img_path)
        except:
            print('Can\'t read'+i.split()[0])
            continue
        if img is None:
            continue
        size = (img.shape[1],img.shape[0])
        del img
        temp = formula[int(i.split()[1])].replace('\\n','')
        # token = set()
        for j in temp.split():
            # token.add(j)
            vocab_temp.add(j)
        data[i.split()[0]] = {'img_path':img_path,'size':size,
        'caption':temp,'caption_len':len(temp.split())+2}#这里需要加上开始以及停止符
        # data[i.split()[0]] = {'img_path':path + 'images/images_' + mode + '/' + i.split()[0],
        # 'token':list(token),'caption':temp,'caption_len':len(temp.split())+2}#这里需要加上开始以及停止符
    vocab_temp = list(vocab_temp)
    vocab = {}
    
    # 确保索引唯一
    vocab['<pad>'] = 0
    vocab['<start>'] = 1
    vocab['<end>'] = 2
    vocab['<unk>'] = 3
    
    # 添加其他词汇，从4开始避免与特殊标记冲突
    for i, token in enumerate(sorted(vocab_temp)):
        vocab[token] = i + 4
    
    return vocab, data


def generate_vocab_for_dataset(data_name, use_huggingface=False, hf_repo='linxy/LaTeX_OCR'):
    """
    为指定数据集生成完整的词汇表
    
    Args:
        data_name: 数据集名称 (如 'small', 'full', 'synthetic_handwrite' 等)
        use_huggingface: 是否使用HuggingFace数据集
        hf_repo: HuggingFace仓库名称
    
    Returns:
        完整的词汇表字典
    """
    
    if use_huggingface:
        print(f"正在从HuggingFace生成 {data_name} 数据集的词汇表...")
        vocab, train_data, val_data = load_huggingface_dataset(hf_repo, data_name)
        return vocab
    else:
        # 原有的本地数据集处理逻辑
        data_path = f'./data/{data_name}/'
        
        # 检查路径是否存在
        if not os.path.exists(data_path):
            print(f"数据路径不存在: {data_path}")
            return None
        
        print(f"正在为 {data_name} 数据集生成词汇表...")
        
        # 生成训练集的词汇表和数据
        try:
            vocab_train, data_train = get_latex_ocrdata(data_path, mode='train')
            print(f"训练集: {len(data_train)} 个样本, {len(vocab_train)} 个词汇")
        except Exception as e:
            print(f"处理训练集时出错: {e}")
            return None
        
        # 生成验证集的数据，但合并词汇表
        try:
            vocab_val, data_val = get_latex_ocrdata(data_path, mode='val')
            print(f"验证集: {len(data_val)} 个样本, {len(vocab_val)} 个词汇")
            
            # 合并词汇表
            all_tokens = set(list(vocab_train.keys()) + list(vocab_val.keys()))
            # 移除特殊标记，我们会单独处理
            all_tokens.discard('<pad>')
            all_tokens.discard('<unk>')
            all_tokens.discard('<start>')
            all_tokens.discard('<end>')
            
            vocab = {}
            # 首先添加特殊标记
            vocab['<pad>'] = 0
            vocab['<start>'] = 1
            vocab['<end>'] = 2
            vocab['<unk>'] = 3
            
            # 然后添加其他词汇
            for i, token in enumerate(sorted(all_tokens)):
                vocab[token] = i + 4  # 从4开始，因为0-3被特殊标记占用
            
            print(f"最终词汇表大小: {len(vocab)}")
            
        except Exception as e:
            print(f"处理验证集时出错: {e}")
            # 如果验证集出错，只使用训练集的词汇表
            vocab = vocab_train
            data_val = {}
        
        # 生成测试集数据（如果存在）
        data_test = {}
        try:
            vocab_test, data_test = get_latex_ocrdata(data_path, mode='test')
            print(f"测试集: {len(data_test)} 个样本")
        except Exception as e:
            print(f"处理测试集时出错: {e}")
        
        # 保存词汇表
        vocab_path = os.path.join(data_path, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
        print(f"词汇表已保存到: {vocab_path}")
        
        # 保存数据文件
        # 如果存在分离的训练/验证文件，分别保存
        if os.path.exists(os.path.join(data_path, 'matching/train.matching.txt')):
            train_path = os.path.join(data_path, 'train.json')
            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump(data_train, f, indent=2, ensure_ascii=False)
            print(f"训练数据已保存到: {train_path}")
        
        if os.path.exists(os.path.join(data_path, 'matching/val.matching.txt')):
            val_path = os.path.join(data_path, 'val.json')
            with open(val_path, 'w', encoding='utf-8') as f:
                json.dump(data_val, f, indent=2, ensure_ascii=False)
            print(f"验证数据已保存到: {val_path}")
        
        if data_test and os.path.exists(os.path.join(data_path, 'matching/test.matching.txt')):
            test_path = os.path.join(data_path, 'test.json')
            with open(test_path, 'w', encoding='utf-8') as f:
                json.dump(data_test, f, indent=2, ensure_ascii=False)
            print(f"测试数据已保存到: {test_path}")
        
        # 如果没有分离文件，保存为单个data.json
        if not os.path.exists(os.path.join(data_path, 'train.json')):
            all_data = {**data_train, **data_val, **data_test}
            data_path_json = os.path.join(data_path, 'data.json')
            with open(data_path_json, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            print(f"所有数据已保存到: {data_path_json}")
        
        return vocab


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.
    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.
    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    梯度裁剪用于避免梯度爆炸
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

 
def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
    decoder_optimizer, score, is_best, use_huggingface=False, vocab_size=None, keep_checkpoints=5):
    """
    Saves model checkpoint and automatically removes old checkpoints to save disk space.
    Keeps the last N checkpoints + 1 best checkpoint separately.
    Also creates a tarball of the checkpoints directory after saving.
    
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param score: validation score for this epoch
    :param is_best: is this checkpoint the best so far?
    :param use_huggingface: whether using HuggingFace dataset
    :param vocab_size: vocabulary size (for validation when loading checkpoint)
    :param keep_checkpoints: number of recent checkpoints to keep (default: 5)
    """
    # 创建checkpoints文件夹
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 根据数据源添加标识前缀
    source_prefix = 'hf' if use_huggingface else 'local'
    
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'score': score,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer,
             'use_huggingface': use_huggingface,
             'vocab_size': vocab_size}
    
    # 保存当前checkpoint（带epoch信息）
    filename = os.path.join(checkpoint_dir, f'checkpoint_{source_prefix}_{data_name}_epoch{epoch}.pth.tar')
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")
    
    # 如果是最佳checkpoint，单独保存
    if is_best:
        best_filename = os.path.join(checkpoint_dir, f'BEST_checkpoint_{source_prefix}_{data_name}.pth.tar')
        torch.save(state, best_filename)
        print(f"Best checkpoint saved: {best_filename}")
    
    # 删除旧的checkpoint，只保留最近的N个
    _cleanup_old_checkpoints(checkpoint_dir, source_prefix, data_name, keep_checkpoints)
    
    # 自动打包checkpoints文件夹
    _create_checkpoint_tarball(checkpoint_dir)


def _cleanup_old_checkpoints(checkpoint_dir, source_prefix, data_name, keep_checkpoints):
    """
    删除旧的checkpoint文件，只保留最近的N个（不删除BEST checkpoint）
    
    :param checkpoint_dir: checkpoint目录
    :param source_prefix: 数据源前缀 ('hf' 或 'local')
    :param data_name: 数据集名称
    :param keep_checkpoints: 保留的checkpoint数量
    """
    import glob
    import re
    
    # 查找所有当前数据集对应的checkpoint文件（不包括BEST）
    pattern = os.path.join(checkpoint_dir, f'checkpoint_{source_prefix}_{data_name}_epoch*.pth.tar')
    checkpoints = glob.glob(pattern)
    
    if len(checkpoints) <= keep_checkpoints:
        return
    
    # 按epoch号排序（提取数字排序，而不是字典序）
    def extract_epoch_num(filepath):
        match = re.search(r'epoch(\d+)', filepath)
        return int(match.group(1)) if match else 0
    
    checkpoints_sorted = sorted(checkpoints, key=extract_epoch_num)
    
    # 如果checkpoint数量超过限制，删除最旧的
    num_to_delete = len(checkpoints_sorted) - keep_checkpoints
    for i in range(num_to_delete):
        old_checkpoint = checkpoints_sorted[i]
        try:
            os.remove(old_checkpoint)
            print(f"Removed old checkpoint: {old_checkpoint}")
        except Exception as e:
            print(f"Failed to remove checkpoint {old_checkpoint}: {e}")


def _create_checkpoint_tarball(checkpoint_dir):
    """
    创建checkpoints目录的压缩包，用于Kaggle环境下保存训练结果
    
    :param checkpoint_dir: checkpoint目录
    """
    import tarfile
    import time
    
    try:
        # 生成带时间戳的tarball文件名，避免被覆盖
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        tarball_name = f'checkpoints_{timestamp}.tar.gz'
        
        # 创建压缩包
        with tarfile.open(tarball_name, 'w:gz') as tar:
            tar.add(checkpoint_dir, arcname=os.path.basename(checkpoint_dir))
        
        print(f"Checkpoints packed successfully: {tarball_name}")
        
        # 同时创建一个最新的压缩包（固定名称，方便下载）
        latest_tarball = 'checkpoints.tar.gz'
        with tarfile.open(latest_tarball, 'w:gz') as tar:
            tar.add(checkpoint_dir, arcname=os.path.basename(checkpoint_dir))
        
        print(f"Latest checkpoints pack updated: {latest_tarball}")
        
    except Exception as e:
        print(f"Warning: Failed to create checkpoint tarball: {e}")


class AverageMeter(object):
    """
    一个用于跟踪变量当前值，平均值，和以及计数的对象
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)