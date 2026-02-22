from typing import Tuple, Optional, Union, List, Literal
import torch.nn.functional as F
import torch
from torch import Tensor
import torch.nn as nn
import argparse
from pylot.metrics.util import _metric_reduction, _inputs_as_onehot
from pylot.loss.segmentation import soft_dice_loss
import numpy as np
from skimage.morphology import skeletonize, dilation
from typing import Callable
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Connectivity Loss
# -----------------------------------------------------------------------------

def connectivity_matrix(multimask, class_num):
    ##### converting segmentation masks to connectivity masks ####

    [batch, _, rows, cols] = multimask.shape
    # batch = 1
    conn = torch.zeros([batch, class_num * 8, rows, cols]).cuda()
    for i in range(class_num):
        mask = multimask[:, i, :, :]
        # print(mask.shape)
        up = torch.zeros([batch, rows, cols]).cuda()  # move the orignal mask to up
        down = torch.zeros([batch, rows, cols]).cuda()
        left = torch.zeros([batch, rows, cols]).cuda()
        right = torch.zeros([batch, rows, cols]).cuda()
        up_left = torch.zeros([batch, rows, cols]).cuda()
        up_right = torch.zeros([batch, rows, cols]).cuda()
        down_left = torch.zeros([batch, rows, cols]).cuda()
        down_right = torch.zeros([batch, rows, cols]).cuda()

        up[:, :rows - 1, :] = mask[:, 1:rows, :]
        down[:, 1:rows, :] = mask[:, 0:rows - 1, :]
        left[:, :, :cols - 1] = mask[:, :, 1:cols]
        right[:, :, 1:cols] = mask[:, :, :cols - 1]
        up_left[:, 0:rows - 1, 0:cols - 1] = mask[:, 1:rows, 1:cols]
        up_right[:, 0:rows - 1, 1:cols] = mask[:, 1:rows, 0:cols - 1]
        down_left[:, 1:rows, 0:cols - 1] = mask[:, 0:rows - 1, 1:cols]
        down_right[:, 1:rows, 1:cols] = mask[:, 0:rows - 1, 0:cols - 1]

        conn[:, (i * 8) + 0, :, :] = mask * down_right
        conn[:, (i * 8) + 1, :, :] = mask * down
        conn[:, (i * 8) + 2, :, :] = mask * down_left
        conn[:, (i * 8) + 3, :, :] = mask * right
        conn[:, (i * 8) + 4, :, :] = mask * left
        conn[:, (i * 8) + 5, :, :] = mask * up_right
        conn[:, (i * 8) + 6, :, :] = mask * up
        conn[:, (i * 8) + 7, :, :] = mask * up_left

    conn = conn.float()
    conn = conn.squeeze()
    # print(conn.shape)
    return conn


def Bilateral_voting(c_map, hori_translation, verti_translation):
    ####  bilateral voting and convert connectivity-based output into segmentation map. ####

    batch, class_num, channel, row, column = c_map.size()
    vote_out = torch.zeros([batch, class_num, channel, row, column]).cuda()
    # print(vote_out.shape,c_map.shape,hori_translation.shape)
    right = (
        torch.bmm(c_map[:, :, 4].contiguous().view(-1, row, column), hori_translation.view(-1, column, column))).view(
        batch, class_num, row, column)

    left = (torch.bmm(c_map[:, :, 3].contiguous().view(-1, row, column),
                      hori_translation.transpose(3, 2).view(-1, column, column))).view(batch, class_num, row, column)

    left_bottom = (torch.bmm(verti_translation.transpose(3, 2).view(-1, row, row),
                             c_map[:, :, 5].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    left_bottom = (
        torch.bmm(left_bottom.view(-1, row, column), hori_translation.transpose(3, 2).view(-1, column, column))).view(
        batch, class_num, row, column)
    right_above = (
        torch.bmm(verti_translation.view(-1, row, row), c_map[:, :, 2].contiguous().view(-1, row, column))).view(batch,
                                                                                                                 class_num,
                                                                                                                 row,
                                                                                                                 column)
    right_above = (torch.bmm(right_above.view(-1, row, column), hori_translation.view(-1, column, column))).view(batch,
                                                                                                                 class_num,
                                                                                                                 row,
                                                                                                                 column)
    left_above = (
        torch.bmm(verti_translation.view(-1, row, row), c_map[:, :, 0].contiguous().view(-1, row, column))).view(batch,
                                                                                                                 class_num,
                                                                                                                 row,
                                                                                                                 column)
    left_above = (
        torch.bmm(left_above.view(-1, row, column), hori_translation.transpose(3, 2).view(-1, column, column))).view(
        batch, class_num, row, column)
    bottom = (torch.bmm(verti_translation.transpose(3, 2).view(-1, row, row),
                        c_map[:, :, 6].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    up = (torch.bmm(verti_translation.view(-1, row, row), c_map[:, :, 1].contiguous().view(-1, row, column))).view(
        batch, class_num, row, column)
    right_bottom = (torch.bmm(verti_translation.transpose(3, 2).view(-1, row, row),
                              c_map[:, :, 7].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    right_bottom = (torch.bmm(right_bottom.view(-1, row, column), hori_translation.view(-1, column, column))).view(
        batch, class_num, row, column)

    vote_out[:, :, 0] = (c_map[:, :, 0]) * (right_bottom)
    vote_out[:, :, 1] = (c_map[:, :, 1]) * (bottom)
    vote_out[:, :, 2] = (c_map[:, :, 2]) * (left_bottom)
    vote_out[:, :, 3] = (c_map[:, :, 3]) * (right)
    vote_out[:, :, 4] = (c_map[:, :, 4]) * (left)
    vote_out[:, :, 5] = (c_map[:, :, 5]) * (right_above)
    vote_out[:, :, 6] = (c_map[:, :, 6]) * (up)
    vote_out[:, :, 7] = (c_map[:, :, 7]) * (left_above)

    pred_mask, _ = torch.max(vote_out, dim=2)
    ###
    # vote_out = vote_out.view(batch,-1, row, column)
    return pred_mask, vote_out


class diceloss(nn.Module):
    def __init__(self, bin_wide, density):
        super(diceloss, self).__init__()
        self.bin_wide = bin_wide
        self.density = density

    def soft_dice_coeff(self, y_pred, y_true, class_i=None):
        smooth = 0.0001  # may change

        i = torch.sum(y_true, dim=(1, 2))
        j = torch.sum(y_pred, dim=(1, 2))
        intersection = torch.sum(y_true * y_pred, dim=(1, 2))

        score = (2. * intersection + smooth) / (i + j + smooth)

        if self.bin_wide:
            weight = density_weight(self.bin_wide[class_i], i, self.density[class_i])
            return (1 - score) * weight
        else:
            return (1 - score)

    def soft_diceloss(self, y_pred, y_true, class_i=None):
        loss = self.soft_dice_coeff(y_true, y_pred, class_i)
        return loss.mean()

    def __call__(self, y_pred, y_true, class_i=None):

        b = self.soft_diceloss(y_true, y_pred, class_i)
        return b


def density_weight(bin_wide, gt_cnt, density):
    index = gt_cnt // bin_wide

    selected_density = [density[index[i].long()] for i in range(gt_cnt.shape[0])]
    selected_density = torch.tensor(selected_density).cuda()
    log_inv_density = torch.log(1 / (selected_density + 0.0001))

    return log_inv_density


class connect_loss(nn.Module):
    def __init__(self, args, hori_translation, verti_translation, density=None, bin_wide=None):
        super(connect_loss, self).__init__()
        #self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.BCEloss = nn.BCELoss(reduction='none')
        self.diceloss = diceloss(bin_wide=bin_wide, density=density)
        self.bin_wide = bin_wide
        self.density = density
        self.args = args

        self.verti_translation = verti_translation
        self.hori_translation = hori_translation

    def edge_loss(self, vote_out, edge):
        pred_mask_min, _ = torch.min(vote_out.cuda(), dim=2)
        pred_mask_min = pred_mask_min * edge
        minloss = self.BCEloss(pred_mask_min, torch.full_like(pred_mask_min, 0))
        return (minloss.sum() / pred_mask_min.sum())  # +maxloss

    def forward(self, c_map, target):

        loss = self.single_class_forward(c_map, target)

        return loss


    def single_class_forward(self, c_map, target):
        #######
        ### c_map: (B, 8, H, W), B: batch, C: class number
        ### target: (B, 1, H, W)
        #######

        batch_num = c_map.shape[0]
        target = target.float()

        ### get your connectivity_mask ###
        con_target = connectivity_matrix(target, self.args.num_class)  # (B, 8, H, W)

        ### matrix for shifting
        hori_translation = self.hori_translation.repeat(batch_num, 1, 1, 1).cuda()
        verti_translation = self.verti_translation.repeat(batch_num, 1, 1, 1).cuda()

        c_map = F.sigmoid(c_map)

        ### get edges gt###
        class_conn = con_target.view([c_map.shape[0], self.args.num_class, 8, c_map.shape[2], c_map.shape[3]])
        sum_conn = torch.sum(class_conn, dim=2)

        ## edge: (B, 1, H, W)
        edge = torch.where((sum_conn < 8) & (sum_conn > 0), torch.full_like(sum_conn, 1), torch.full_like(sum_conn, 0))

        ### bilateral voting #####
        ## pred: (B, 1, H, W), bicon_map: (B, 1, 8, H, W)
        class_pred = c_map.view([c_map.shape[0], self.args.num_class, 8, c_map.shape[2], c_map.shape[3]])
        pred, bicon_map = Bilateral_voting(class_pred, hori_translation, verti_translation)

        edge_l = self.edge_loss(bicon_map, edge)

        dice_l = self.diceloss(pred[:, 0], target[:, 0])

        bce_loss = self.BCEloss(pred, target).mean()
        if con_target.shape[0] == 8 and con_target.shape[1] == 512:
            con_target = torch.unsqueeze(con_target, dim=0)
        conn_l = self.BCEloss(c_map, con_target).mean()

        if self.args.dataset == 'chase':
            loss = bce_loss + conn_l + edge_l + dice_l
        else:
            bicon_l = self.BCEloss(bicon_map.squeeze(1), con_target).mean()
            loss = bce_loss + conn_l + edge_l + 0.2 * bicon_l + dice_l  # + bce_loss# +loss_out_dice# +sum_l # + edge_l+loss_out_dice

        return loss


args = argparse.Namespace(num_class=1, dataset="chase")

W = 512
H = 512

hori_translation = torch.zeros([1, 1, W, W])
for i in range(W - 1):
    hori_translation[:, :, i, i + 1] = torch.tensor(1.0)
verti_translation = torch.zeros([1, args.num_class, H, H])
for j in range(H - 1):
    verti_translation[:, :, j, j + 1] = torch.tensor(1.0)
hori_translation = hori_translation.float()
verti_translation = verti_translation.float()

Connectivity_Loss = connect_loss(args, hori_translation, verti_translation)


class SoftSkeletonRecallLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin: Callable = None,
        smooth: float = 1.,
        do_tube: bool = True,
    ):
        """
        A structure-aware loss that computes recall between predicted mask and GT skeleton.
        Inputs:
            - x: raw prediction (logits), shape (B, 1, H, W)
            - y: binary ground truth mask (0 or 1), shape (B, 1, H, W)
        """
        super(SoftSkeletonRecallLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.do_tube = do_tube

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # Step 1: Apply sigmoid if needed
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)  # e.g., torch.sigmoid

        # Step 2: Detach y and convert to skeleton (possibly with dilation)
        y_skel = self._skeletonize_batch(y)

        # Step 3: Compute soft recall: intersection / gt
        axes = tuple(range(2, len(x.shape)))  # spatial dims only

        intersection = (x * y_skel).sum(dim=axes)
        gt_sum = y_skel.sum(dim=axes)

        recall = (intersection + self.smooth) / (gt_sum + self.smooth)
        recall = recall.mean()

        return -recall  # max recall -> min loss

    def _skeletonize_batch(self, y: torch.Tensor) -> torch.Tensor:
        """
        Applies skeletonization + optional dilation to binary masks (B x 1 x H x W),
        returns a tensor of the same shape.
        """
        y_np = y.detach().cpu().numpy()
        B, C, H, W = y_np.shape
        y_skel = np.zeros_like(y_np, dtype=np.float32)

        for b in range(B):
            for c in range(C):
                mask = y_np[b, c]
                if np.any(mask):  # skip empty masks
                    skel = skeletonize(mask > 0.5)
                    if self.do_tube:
                        skel = dilation(skel)
                        skel = dilation(skel)
                    y_skel[b, c] = skel.astype(np.float32)

        return torch.from_numpy(y_skel).to(y.device)


skeleton_loss_term = SoftSkeletonRecallLoss(
    apply_nonlin=torch.sigmoid,
    do_tube=True
)


class SoftSkeletonize(nn.Module):
    def __init__(self, num_iter=10):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), stride=(1, 1), padding=(1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), stride=(1, 1), padding=(0, 1))
            return torch.min(p1, p2)
        else:
            raise ValueError("Unsupported tensor shape for soft_erode")

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), stride=(1, 1), padding=(1, 1))
        else:
            raise ValueError("Unsupported tensor shape for soft_dilate")

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)

        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):
        return self.soft_skel(img)


class SoftCLDiceLoss(nn.Module):
    def __init__(self, num_iter=3, smooth=1.0, exclude_background=False, visualize=False):
        super(SoftCLDiceLoss, self).__init__()
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=num_iter)
        self.exclude_background = exclude_background
        self.visualize = visualize

    def forward(self, y_pred_logits, y_true):

        y_pred = torch.sigmoid(y_pred_logits)

        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]

        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)

        if self.visualize:
            with torch.no_grad():
                skel_np = skel_true[0, 0].detach().cpu().numpy()
                plt.imshow(skel_np, cmap='gray')
                plt.title("Skeleton of Ground Truth")
                plt.axis("off")
                plt.show()

        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)

        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


softcl_loss_term = SoftCLDiceLoss(num_iter=10, visualize=False)


class FocalDiceLoss(nn.Module):
    """
    Focal Loss + Soft Dice Loss
    """
    def __init__(self, from_logits: bool = False, gamma: float = 20.0, batch_reduction: Optional[Literal["mean"]] = None, **kwargs):
        super().__init__()
        self.batch_reduction = batch_reduction
        self.from_logits = from_logits
        self.gamma = gamma
        self.kwargs = kwargs

    def __call__(self, y_pred, y_true, ):
        # y_pred shape: B x 1 x H x W
        # We are doing binary segmentation so channel = 1


        focal_loss_term = focal_loss(y_pred, y_true,
                                     gamma=self.gamma,
                                     reduction='mean',
                                     batch_reduction=None,
                                     from_logits=self.from_logits,
                                     )

        dice_loss_term = soft_dice_loss(y_pred, y_true,
                              mode='binary',
                              weights=None,
                              reduction='mean', # there's only 1 channel so this is fine
                              batch_reduction=None,
                              from_logits=self.from_logits,
                              **self.kwargs
                              )
        # aaa = softcl_loss_term(y_pred, y_true)
        loss = focal_loss_term + dice_loss_term #+ 0.5 * skeleton_loss_term(y_pred, y_true)
        # loss = focal_loss_term + dice_loss_term + aaa
        # print(aaa)
        # print("-------------------")

        if self.batch_reduction == 'mean':
            return loss.mean()
        else:
            return loss

        # loss = Connectivity_Loss(y_pred, y_true)
        #
        # return loss

# -----------------------------------------------------------------------------
# Focal Loss
# -----------------------------------------------------------------------------

def focal_loss(
    y_pred: Tensor,
    y_true: Tensor,
    gamma: float = 20.0,
    weights: Optional[Tensor] = None,
    channel_weights: Optional[Tensor] = None,
    mode: str = "auto",
    reduction: str = "mean", # Reduction over channels
    batch_reduction: str = "mean", 
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
    eps: float = 1e-7
) -> Tensor:
    """
    Binary focall loss that allows per-pixel weights
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
    """
    if weights is not None:
        batch_size, num_classes = y_pred.shape[:2]
        weights = weights.view(batch_size, num_classes, -1)

    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits
    )

    loss = - binary_focal_cross_entropy(
        y_pred, y_true, weights=weights, eps=eps, dim=-1, gamma=gamma
    )

    batch_loss = _metric_reduction(
        loss, reduction=reduction, batch_reduction=batch_reduction, weights=channel_weights, ignore_index=ignore_index
    )

    return batch_loss


def binary_focal_cross_entropy(
    y_pred: Tensor,
    y_true: Tensor,
    weights: Optional[Tensor] = None,
    gamma: float = 20.0,
    eps: float = 1e-7,
    dim = None,
    ):
    """
    Returns -binary focal loss
    https://focal-loss.readthedocs.io/en/latest/generated/focal_loss.binary_focal_loss.html#focal_loss.binary_focal_loss
    """
    assert y_pred.shape == y_true.shape, f"y_pred.shape {y_pred.shape} != y_true.shape {y_true.shape}"
    if weights is not None:
        assert y_pred.shape == weights.shape, f"y_pred.shape={y_pred.shape}, weights.shape={weights.shape} do not match"

    left_term = y_true * torch.log(y_pred + eps) * (1 - y_pred)**gamma
    right_term = (1 - y_true) * torch.log(1-y_pred + eps) * y_pred**gamma

    if weights is not None:
        return torch.mean((left_term + right_term)*weights, dim=dim)
    else:
        return torch.mean(left_term + right_term, dim=dim)
    


