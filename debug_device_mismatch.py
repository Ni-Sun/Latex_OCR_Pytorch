# -*- coding: utf-8 -*-
"""
设备不匹配诊断脚本
用于检查模型、优化器和张量是否在同一设备上
"""

import torch
import torch.nn as nn


def check_model_device(model, device_name='GPU'):
    """检查模型所有参数是否在同一设备上"""
    devices = set()
    for param in model.parameters():
        devices.add(param.device)
    
    if len(devices) == 1:
        print(f"✓ {model.__class__.__name__}: 所有参数在同一设备上 - {list(devices)[0]}")
        return True
    else:
        print(f"✗ {model.__class__.__name__}: 参数分布在多个设备上:")
        for device in devices:
            param_count = sum(1 for p in model.parameters() if p.device == device)
            print(f"    {device}: {param_count} 个参数")
        return False


def check_optimizer_device(optimizer, device_name='GPU'):
    """检查优化器的状态是否在正确的设备上"""
    issues = []
    
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param_device = param.device
            
            # 检查优化器状态
            param_id = id(param)
            if param_id in optimizer.state:
                state = optimizer.state[param_id]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        if value.device != param_device:
                            issues.append(f"参数在 {param_device}，但状态'{key}'在 {value.device}")
    
    if issues:
        print(f"✗ {optimizer.__class__.__name__} 有设备不匹配的问题:")
        for issue in issues[:5]:  # 只显示前5个
            print(f"    {issue}")
        if len(issues) > 5:
            print(f"    ... 还有 {len(issues)-5} 个问题")
        return False
    else:
        print(f"✓ {optimizer.__class__.__name__}: 所有状态与参数设备匹配")
        return True


def diagnose_training_setup(encoder, decoder, encoder_optimizer, decoder_optimizer, device):
    """诊断训练设置中的设备问题"""
    print("\n" + "="*60)
    print("设备匹配诊断报告")
    print("="*60)
    
    all_ok = True
    
    # 检查模型
    print("\n[模型检查]")
    all_ok &= check_model_device(encoder)
    all_ok &= check_model_device(decoder)
    
    # 检查优化器
    print("\n[优化器检查]")
    all_ok &= check_optimizer_device(encoder_optimizer)
    all_ok &= check_optimizer_device(decoder_optimizer)
    
    # 检查目标设备
    print(f"\n[目标设备]")
    print(f"配置的目标设备: {device}")
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ 诊断结果: 全部正常！")
    else:
        print("✗ 诊断结果: 发现设备不匹配问题！")
    print("="*60 + "\n")
    
    return all_ok


def move_optimizer_to_device(optimizer, device):
    """
    将优化器的状态移到指定设备
    这在从checkpoint加载优化器时特别重要
    """
    print(f"正在将优化器状态移到 {device}...")
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    print(f"✓ 优化器状态已移到 {device}")


if __name__ == '__main__':
    print("设备不匹配诊断工具")
    print("=" * 60)
    print("\n使用方法:")
    print("1. 在 train.py 中导入此模块")
    print("2. 在训练前调用: diagnose_training_setup(encoder, decoder, encoder_optimizer, decoder_optimizer, device)")
    print("3. 如果发现问题，调用: move_optimizer_to_device(optimizer, device)")
    print("\n" + "=" * 60)
