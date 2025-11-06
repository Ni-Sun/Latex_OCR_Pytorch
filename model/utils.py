import os
import numpy as np
import json
import cv2
import torch

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


def generate_vocab_for_dataset(data_name):
    """
    为指定数据集生成完整的词汇表，合并训练集和验证集的词汇
    
    Args:
        data_name: 数据集名称 (如 'small', 'full', 'hand' 等)
    
    Returns:
        完整的词汇表字典
    """
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
    decoder_optimizer,score, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'score': score,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer':encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


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