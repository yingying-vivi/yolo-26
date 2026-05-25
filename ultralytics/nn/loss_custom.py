# ultralytics/nn/loss_custom.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# 1️⃣ IoU Loss（CIoU）
# -------------------------------
def bbox_iou(box1, box2, eps=1e-7):
    # box format: (x1, y1, x2, y2)
    inter = (torch.min(box1[..., 2:], box2[..., 2:]) - torch.max(box1[..., :2], box2[..., :2])).clamp(0).prod(2)

    area1 = (box1[..., 2:] - box1[..., :2]).prod(2)
    area2 = (box2[..., 2:] - box2[..., :2]).prod(2)

    union = area1 + area2 - inter + eps
    iou = inter / union

    return iou


def ciou_loss(pred, target):
    iou = bbox_iou(pred, target)
    return 1 - iou


# -------------------------------
# 2️⃣ Boundary Loss（核心创新）
# -------------------------------
def boundary_loss(pred_mask, gt_mask):
    # Sobel边缘提取
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).to(pred_mask.device)

    sobel_y = sobel_x.t()

    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    pred_edge = F.conv2d(pred_mask, sobel_x, padding=1) + F.conv2d(pred_mask, sobel_y, padding=1)

    gt_edge = F.conv2d(gt_mask, sobel_x, padding=1) + F.conv2d(gt_mask, sobel_y, padding=1)

    return F.l1_loss(pred_edge, gt_edge)


# -------------------------------
# 3️⃣ Focal Loss（分类优化）
# -------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob = torch.sigmoid(pred)
        pt = target * prob + (1 - target) * (1 - prob)

        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


# -------------------------------
# 4️⃣ Small Object Weight（小目标增强）
# -------------------------------
def small_object_weight(box):
    # box area
    area = (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])
    weight = 1.0 / (area + 1e-6)
    return weight.detach()


# -------------------------------
# 5️⃣ ProgLoss（动态权重）
# -------------------------------
class ProgLoss:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def weight(self, epoch):
        return 1 - epoch / self.total_epochs


# -------------------------------
# 6️⃣ 总Loss（最终版本）
# -------------------------------
class CustomYOLOLoss(nn.Module):
    def __init__(self, total_epochs=300):
        super().__init__()
        self.focal = FocalLoss()
        self.prog = ProgLoss(total_epochs)

    def forward(self, preds, targets, masks=None, epoch=0):

        pred_box, pred_cls = preds
        target_box, target_cls = targets

        # ----------------
        # Box Loss
        # ----------------
        iou_loss_val = ciou_loss(pred_box, target_box)

        # 小目标权重
        weight = small_object_weight(target_box)
        iou_loss_val = (iou_loss_val * weight).mean()

        # ----------------
        # 分类 Loss
        # ----------------
        cls_loss = self.focal(pred_cls, target_cls)

        # ----------------
        # Boundary Loss
        # ----------------
        if masks is not None:
            pred_mask, gt_mask = masks
            b_loss = boundary_loss(pred_mask, gt_mask)
        else:
            b_loss = 0

        # ----------------
        # ProgLoss 权重
        # ----------------
        prog_w = self.prog.weight(epoch)

        total_loss = prog_w * (iou_loss_val + cls_loss) + (1 - prog_w) * (iou_loss_val + cls_loss + b_loss)

        return total_loss, {"iou": iou_loss_val, "cls": cls_loss, "boundary": b_loss}
