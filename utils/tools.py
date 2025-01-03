import os


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def get_model_summary(model, device, input_shape=(1, 3, 704, 704), detailed=True):
    """
    获取模型的统计信息，包括参数量和计算量

    Args:
        model: 要分析的模型
        device: 运行设备 (cuda:id 或 cpu)
        input_shape: 输入张量的形状，默认为 (1, 3, 704, 704)
        detailed: 是否输出详细的每个模块的信息

    Returns:
        str: 包含所有统计信息的字符串
    """
    from thop import profile
    import io
    from contextlib import redirect_stdout
    import torch
    import copy

    def params_to_string(params_num):
        """将参数数量转换为更易读的形式"""
        if params_num >= 1e6:
            return f'{params_num / 1e6:.2f}M'
        elif params_num >= 1e3:
            return f'{params_num / 1e3:.2f}K'
        return str(params_num)

    # 创建模型的深拷贝并移动到对应设备
    model_copy = copy.deepcopy(model).to(device)

    output = io.StringIO()
    with redirect_stdout(output):
        # 详细模式下输出每个模块的参数量
        if detailed:
            print("Detailed parameters for each module:")
            print("-" * 50)
            for name, module in model_copy.named_children():
                params = sum(p.numel() for p in module.parameters())
                print(f'{name}: {params_to_string(params)} parameters')
            print("-" * 50)

        # 计算总参数量
        total_params = sum(p.numel() for p in model_copy.parameters())
        trainable_params = sum(p.numel() for p in model_copy.parameters() if p.requires_grad)
        print(f'Total Parameters: {params_to_string(total_params)}')
        print(f'Trainable Parameters: {params_to_string(trainable_params)}')

        # thop计算FLOPs和参数量
        input_tensor = torch.randn(*input_shape).to(device)
        flops, thop_params = profile(model_copy, inputs=(input_tensor,))
        print(f'THOP Parameters: {params_to_string(thop_params)}')
        print(f'FLOPs: {flops / 1e9:.2f}G')

    summary_str = output.getvalue()

    # 清理model_copy
    del model_copy
    return summary_str

def save_model_summary(model, exp_dir: str, device: str):
    """保存模型统计信息到实验目录"""
    # 获取模型统计信息
    summary = get_model_summary(model, device)

    # 保存到文件
    with open(os.path.join(exp_dir, 'model_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
