# -*- coding: utf-8 -*-
import time
from config import *
import os
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from model.utils import *
from model import metrics,dataloader,model
from torch.utils.checkpoint import checkpoint as train_ck

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model.device = device
'''
如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
'''
cudnn.benchmark = True

# 初始化日志记录器（日志将保存到 ./log 目录，带时间戳的文件名）
logger = setup_logger('./log')


def main():
    """
    Training and validation.
    """

    global best_score, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, keep_checkpoints

    logger.info("="*80)
    logger.info("Starting training session")
    logger.info(f"Data name: {data_name}")
    logger.info(f"Use HuggingFace: {use_huggingface}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Max epochs per run: {max_epochs_per_run if max_epochs_per_run else 'unlimited'}")
    logger.info("="*80)

    # 检查是否使用HuggingFace数据集并且数据不存在，则先加载数据集
    if use_huggingface:
        if not os.path.exists(vocab_path):
            logger.info("检测到使用HuggingFace数据集，正在下载和预处理数据...")
            from model.utils import load_huggingface_dataset
            vocab, train_data, val_data = load_huggingface_dataset(hf_repo, data_name)
            if vocab is None:
                logger.error("数据集加载失败，请检查网络连接和数据集名称")
                return
            logger.info("数据集下载和预处理完成！")

    # 字典文件
    word_map = load_json(vocab_path)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = model.DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = model.Encoder()
        # encoder_optimizer = None
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr)

    else:
        checkpoint = torch.load(checkpoint, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_score = checkpoint['score']
        decoder = checkpoint['decoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        logger.info(f"Best score so far: {best_score}")
        logger.info(f"Epochs since improvement: {epochs_since_improvement}")
        
        # 验证 checkpoint 中的数据源和词汇表是否匹配
        checkpoint_use_hf = checkpoint.get('use_huggingface', None)
        checkpoint_vocab_size = checkpoint.get('vocab_size', None)
        
        # 检查数据源是否一致
        if checkpoint_use_hf is not None and checkpoint_use_hf != use_huggingface:
            logger.warning(f"Checkpoint 与当前数据源不匹配!")
            logger.warning(f"  Checkpoint 数据源: {'HuggingFace' if checkpoint_use_hf else '本地'}")
            logger.warning(f"  当前数据源: {'HuggingFace' if use_huggingface else '本地'}")
            logger.warning(f"  建议使用匹配的数据源重新加载 checkpoint")
        
        # 检查词汇表大小是否一致
        if checkpoint_vocab_size is not None and checkpoint_vocab_size != len(word_map):
            logger.error(f"Checkpoint 词汇表大小不匹配!")
            logger.error(f"  Checkpoint 词汇表大小: {checkpoint_vocab_size}")
            logger.error(f"  当前词汇表大小: {len(word_map)}")
            logger.error(f"  请使用相同数据源的词汇表或重新生成模型")
            raise ValueError("Checkpoint 词汇表大小不匹配，无法继续训练")
        # encoder_optimizer = checkpoint['encoder_optimizer']
        # encoder_optimizer = None
        # if fine_tune_encoder is True and encoder_optimizer is None:
        #     encoder.fine_tune(fine_tune_encoder)
        #     encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
        #                                          lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    
    # 将优化器的状态也移到GPU上
    # 这是必要的，因为优化器包含的状态张量（如momentum）也需要在正确的设备上
    if checkpoint is not None:
        # 只有从checkpoint加载时才需要这步，因为新创建的优化器已经自动在正确的设备上
        for state in encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        logger.info(f"Optimizer states moved to {device}")

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 自定义的数据集
    train_loader = dataloader.formuladataset(train_set_path,batch_size = batch_size,ratio = 5)
    val_loader = dataloader.formuladataset(val_set_path,batch_size = test_batch_size,ratio = 5)

    # #统计验证集的词频
    # words_freq = cal_word_freq(word_map,val_loader)
    # print(words_freq)
    p = 1#teacher forcing概率
    
    # 计算本次运行的结束epoch
    if max_epochs_per_run is not None:
        end_epoch = start_epoch + max_epochs_per_run
        logger.info(f"Training from epoch {start_epoch} to {end_epoch-1} (max_epochs_per_run={max_epochs_per_run})")
    else:
        end_epoch = epochs
        logger.info(f"Training from epoch {start_epoch} to {end_epoch-1}")
    
    # Epochs
    for epoch in range(start_epoch, min(end_epoch, epochs)):
        train_loader.shuffle()
        val_loader.shuffle()
        #每2个epoch衰减一次teahcer forcing的概率
        if p > 0.05:
            if (epoch % 3 == 0 and epoch != 0):
                p *= 0.9
        else:
            p = 0
        logger.info(f'Start epoch: {epoch}, teacher forcing p: {p:.2f}')

        # 如果迭代4次后没有改善,则对学习率进行衰减,如果迭代20次都没有改善则触发早停.直到最大迭代次数
        if epochs_since_improvement == 70:
            logger.info("Early stopping triggered: 70 epochs without improvement")
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            adjust_learning_rate(decoder_optimizer, 0.7)
            adjust_learning_rate(encoder_optimizer, 0.8)
        #动态学习率调节
        # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, 
        #     patience=4, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=decoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,p=p)#encoder_optimizer=encoder_optimizer,

        # One epoch's validation
        recent_score = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)
        
        # Check if there was an improvement（总是检查改进，不依赖于p值）
        is_best = recent_score > best_score
        best_score = max(recent_score, best_score)
        
        if (p==0):
            logger.info('Teacher forcing stopped!')
            if not is_best:
                epochs_since_improvement += 1
                logger.info(f"Epochs since last improvement: {epochs_since_improvement}")
            else:
                logger.info(f'New Best Score! ({best_score})')
                epochs_since_improvement = 0

        # 保存checkpoint的逻辑：
        # 1. 如果是最佳checkpoint，总是保存
        # 2. 或者每save_freq个epoch保存一次定期checkpoint
        # 3. 当停止teacher forcing后，每save_freq个epoch保存一次
        should_save = is_best or (epoch % save_freq == 0)
        
        if should_save:
            logger.info('Saving checkpoint...')
            save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder,encoder_optimizer,
                        decoder_optimizer, recent_score, is_best, use_huggingface=use_huggingface, 
                        vocab_size=len(word_map), keep_checkpoints=keep_checkpoints)
        
        logger.info('--------------------------------------------------------------------------')
    
    logger.info("="*80)
    logger.info(f"Training session completed at epoch {epoch}")
    logger.info(f"Best score achieved: {best_score}")
    logger.info("="*80)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer,decoder_optimizer, epoch, p):
    """
    Performs one epoch's training.
    :param train_loader: 训练集的dataloader
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: 损失函数
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss (per word decoded)
    top3accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    # for i, (imgs, caps, caplens) in tqdm(enumerate(train_loader)):
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        # try:
        #     imgs = encoder(imgs)
        #     scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        # except:
        # imgs.requires_grad = True
        # imgs = train_ck(encoder,imgs)
        try:
            imgs = encoder(imgs)
        except:
            imgs = train_ck(encoder,imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, p=p)

        # 由于加入开始符<start>以及停止符<end>,caption从第二位开始,知道结束符
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        # targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        scores = scores.to(device)
        loss = criterion(scores, targets)

        # 加入 doubly stochastic attention 正则化
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # 反向传播
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            # if encoder_optimizer is not None:
            #     clip_gradient(encoder_optimizer, grad_clip)

        # 更新权重
        decoder_optimizer.step()
        encoder_optimizer.step()
        # if encoder_optimizer is not None:
        #     encoder_optimizer.step()

        # Keep track of metrics
        top3 = accuracy(scores, targets, 3)
        losses.update(loss.item(), sum(decode_lengths))
        top3accs.update(top3, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                  'Top-3 Accuracy {top3.val:.3f} ({top3.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          loss=losses,
                                                                          top3=top3accs)
            print(msg)
            logger.info(msg)
        # if i % save_freq == 0:
        #     save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder,encoder_optimizer,
        #                 decoder_optimizer, 0,0)
        del imgs, scores, caps_sorted, decode_lengths, alphas, sort_ind, loss, targets
        torch.cuda.empty_cache()


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: 用于验证集的dataloader
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: 损失函数
    :return: 验证集上的BLEU-4 score
    """
    decoder.eval()  # 推断模式,取消dropout以及批标准化
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top3accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        # Batches
        # for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
        # for i, (imgs, caps, caplens) in tqdm(enumerate(val_loader)):
        for i, (imgs, caps, caplens) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, p=0)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top3 = accuracy(scores, targets, 3)
            top3accs.update(top3, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                msg = 'Validation: [{0}/{1}],' \
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f}),' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f}),' \
                      'Top-3 Accuracy {top3.val:.3f} ({top3.avg:.3f}),'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top3=top3accs)
                print(msg)
                logger.info(msg)

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            # for j in range(allcaps.shape[0]):
            #     img_caps = allcaps[j].tolist()
            #     img_captions = list(
            #         map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
            #             img_caps))  # remove <start> and pads
            #     references.append(img_captions)
            caplens = caplens[sort_ind]
            caps = caps[sort_ind]
            for i in range(len(caplens)):
                references.append(caps[i][1:caplens[i]].tolist())
            # Hypotheses
            # 这里直接使用greedy模式进行评价,在推断中一般使用集束搜索模式
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        Score = metrics.evaluate(losses, top3accs, references, hypotheses)
    return Score


if __name__ == '__main__':
    main()