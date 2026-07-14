import torch
import torch.nn as nn
import torch.nn.functional as F


# class SFLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, smoothing=0.1, reduction='mean'):
#         super(SFLoss, self).__init__()
#         self.alpha = alpha  # 类别平衡因子
#         self.gamma = gamma  # 调节困难样本的因子
#         self.smoothing = smoothing  # 标签平滑因子
#         self.reduction = reduction  # 损失聚合方式
#         # self.ce_loss = nn.CrossEntropyLoss(reduction='none')

#     def forward(self, inputs, targets):
#         # 输入: inputs: 模型输出（未经过 softmax 的原始 logits）
#         #      targets: 目标类别标签（整数）

#         # 获取类别数量
#         num_classes = inputs.size(1)

#         # 将 logits 转化为概率（通过 softmax）
#         inputs = F.softmax(inputs, dim=1)

#         # 标签平滑
#         one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
#         smooth_ones = 1.0 - self.smoothing
#         smooth_zeros = self.smoothing / (num_classes - 1)
#         targets_smooth = one_hot * smooth_ones + smooth_zeros

#         # 获取正确类别的平滑后的概率
#         p_t = (inputs * targets_smooth).sum(dim=1, keepdim=True)

#         # 计算 Focal Loss
#         loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t)

#         # # # 计算交叉熵损失
#         # ce_loss = self.ce_loss(inputs, targets)
#         # loss = loss + ce_loss

#         # 损失聚合
#         if self.reduction == 'mean':
#             return loss.mean()  # 返回平均损失
#         elif self.reduction == 'sum':
#             return loss.sum()  # 返回总损失
#         else:
#             return loss  # 不做聚合，返回每个样本的损失


class SFLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, smoothing=0.1, reduction='mean'):
        super(SFLoss, self).__init__()
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', None)

        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        num_classes = inputs.size(1)

        # 标签平滑
        with torch.no_grad():
            one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
            smooth_ones = 1.0 - self.smoothing
            smooth_zeros = self.smoothing / (num_classes - 1)
            targets_smooth = one_hot * smooth_ones + smooth_zeros

        # 计算基础的 Focal Loss
        eps = 1e-8
        focal_weight = (1 - probs) ** self.gamma
        loss_all_classes = -focal_weight * torch.log(probs + eps)
        loss = (targets_smooth * loss_all_classes).sum(dim=1)

        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            loss = loss * alpha_weight

            if self.reduction == 'mean':
                return loss.sum() / (alpha_weight.sum() + eps)

        # 如果没有 alpha，或者 reduction 不是 mean
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
