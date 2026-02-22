import time
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import os
import cv2
import pathlib
import matplotlib.pyplot as plt
import torch.nn as nn
from typing import Literal, Tuple, Optional, Dict, Any, List


# -----------------------------------------------------------------------------
# Params
# -----------------------------------------------------------------------------
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
RES = 512
THRESHOLD = 0.3
file_dir = pathlib.Path(os.path.dirname(__file__))
example_dir = file_dir / "examples"
test_examples = [str(example_dir / x) for x in sorted(os.listdir(example_dir)) if not x.endswith('.npy')]
default_example = test_examples[0]
exp_dir = file_dir / "../checkpoint"
default_model = 'RoadGIE'
model_dict = {'RoadGIE': 'epoch-300.pt'}
ACCUMULATE_CAM = 0


# -----------------------------------------------------------------------------
# DecoderBlock
# -----------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp
        self.deconv1 = nn.Conv2d(in_channels // 4, in_channels // 4, (1, 9), padding=(0, 4))
        self.deconv2 = nn.Conv2d(in_channels // 4, in_channels // 4, (9, 1), padding=(4, 0))
        self.deconv3 = nn.Conv2d(in_channels // 4, in_channels // 4, (9, 1), padding=(4, 0))
        self.deconv4 = nn.Conv2d(in_channels // 4, in_channels // 4, (1, 9), padding=(0, 4))
        self.bn2 = BatchNorm(in_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()
        self._init_weight()

    def forward(self, x):

        x = x.to(dtype=torch.float32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


# -----------------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------------
class DecoderConvWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, do_activation=True):
        super().__init__()
        self.block = DecoderBlock(in_channels, out_channels, BatchNorm=nn.BatchNorm2d, inp=False)

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------------
class Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 do_activation: bool = True,
                 ):
        super(Conv2d, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        lst = [conv]
        if do_activation:
            lst.append(nn.PReLU())
        self.conv = nn.Sequential(*lst)

    def forward(self, x):

        return self.conv(x)


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------
class _UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: List[int] = [64, 64, 64, 64, 64],
                 conv_kernel_size: int = 3,
                 conv: Optional[nn.Module] = None,
                 conv_kwargs: Dict[str, Any] = {}
                 ):

        super(_UNet, self).__init__()

        self.in_channels = in_channels
        padding = (conv_kernel_size - 1) // 2
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for i, feat in enumerate(features):

            if i in [0, 1, 2, 3]:
                self.downs.append(
                    conv(
                        in_channels, feat, kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
                    )
                )
            else:
                self.downs.append(
                    conv(
                        in_channels, feat, kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
                    )
                )
            in_channels = feat

        for i, feat in enumerate(reversed(features)):

            if i in [3]:
                self.ups.append(nn.UpsamplingBilinear2d(scale_factor=2))
                self.ups.append(
                    DecoderConvWrapper(
                        # Factor of 2 is for the skip connections
                        feat * 2, feat, kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
                    )
                )
            else:
                self.ups.append(nn.UpsamplingBilinear2d(scale_factor=2))
                self.ups.append(
                    conv(
                        # Factor of 2 is for the skip connections
                        feat * 2, feat, kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
                    )
                )

        self.bottleneck = conv(
            features[-1], features[-1], kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
        )
        self.final_conv = conv(
            features[0], out_channels, kernel_size=1, padding=0, do_activation=False, **conv_kwargs
        )

        self.gradcam_activation = None

    def forward(self, x: torch.Tensor):

        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
            # if idx == 2:
            #     x.retain_grad()
            #     self.gradcam_activation = x
        logit = self.final_conv(x)

        return logit


class UNet(_UNet):


    def __init__(self, **kwargs) -> None:
        super().__init__(conv=Conv2d, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


# -----------------------------------------------------------------------------
# Preprocess tools
# -----------------------------------------------------------------------------
def bbox_shaded(boxes, shape: Tuple[int, int], device="cpu"):
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.int().cpu().numpy()

    B, N = boxes.shape[0:2]
    out = torch.zeros((B, 1, *shape), device=device)
    for i in range(B):
        for j in range(N):
            x1, y1, x2, y2 = boxes[i, j]
            xmin, xmax = sorted((x1, x2))
            ymin, ymax = sorted((y1, y2))
            out[i, 0, ymin:ymax, xmin:xmax] = 1.0
    return out


def rescale_inputs(inputs: Dict[str, Any], target_size: Tuple[int, int]=(512, 512)):
    h, w = inputs['img'].shape[-2:]
    if (h, w) != target_size:
        scale_h, scale_w = target_size[0] / h, target_size[1] / w
        inputs['img'] = F.interpolate(inputs['img'], size=target_size, mode='bilinear')

        if inputs.get("scribbles") is not None:
            inputs['scribbles'] = F.interpolate(inputs['scribbles'], size=target_size, mode='bilinear')

        if inputs.get("box") is not None:
            coords = inputs["box"].reshape(-1, 2, 2)
            coords[..., 0] *= scale_w
            coords[..., 1] *= scale_h
            inputs["box"] = coords.reshape(1, -1, 4).int()

        if inputs.get("point_coords") is not None:
            coords = inputs["point_coords"].clone()
            coords[..., 0] *= scale_w
            coords[..., 1] *= scale_h
            inputs["point_coords"] = coords.int()
    return inputs


def prepare_inputs(inputs: Dict[str, torch.Tensor], device=None) -> torch.Tensor:

    img = inputs['img']
    if device is None:
        device = img.device

    img = img.to(device)
    shape = tuple(img.shape[-2:])

    image = img.clone()
    img = img[:, 0:1, :, :]

    if inputs.get("box") is not None:
        box_embed = bbox_shaded(inputs['box'], shape=shape, device=device)
    else:
        box_embed = torch.zeros(img.shape, device=device)

    if inputs.get("point_coords") is not None:
        scribble_click_embed = click_onehot(inputs['point_coords'], inputs['point_labels'], shape=shape)
    else:
        scribble_click_embed = torch.zeros((img.shape[0], 2) + shape, device=device)

    if inputs.get("scribbles") is not None:
        scribble_click_embed = torch.clamp(scribble_click_embed + inputs.get('scribbles'), min=0.0, max=1.0)

    if inputs.get('mask_input') is not None:
        mask_input = inputs['mask_input']
    else:
        mask_input = torch.zeros(img.shape, device=img.device)

    x = torch.cat((image, scribble_click_embed, mask_input), dim=-3)

    return x


def click_onehot(point_coords, point_labels, shape: Tuple[int, int], indexing: str = 'xy'):
    device = point_coords.device
    B, N = point_coords.shape[:2]
    embed = torch.zeros((B, 2, *shape), device=device)
    labels = point_labels.flatten().float()

    idx_coords = torch.cat((
        torch.arange(B, device=device).reshape(-1, 1).repeat(1, N)[..., None],
        point_coords
    ), dim=2).reshape(-1, 3)

    if indexing == 'xy':
        embed[idx_coords[:, 0], 0, idx_coords[:, 2], idx_coords[:, 1]] = labels
        embed[idx_coords[:, 0], 1, idx_coords[:, 2], idx_coords[:, 1]] = 1.0 - labels
    else:
        embed[idx_coords[:, 0], 0, idx_coords[:, 1], idx_coords[:, 2]] = labels
        embed[idx_coords[:, 0], 1, idx_coords[:, 1], idx_coords[:, 2]] = 1.0 - labels

    return embed


def compute_gradcam(unet_model: nn.Module, input_tensor: torch.Tensor, target_index: int = 0):

    global ACCUMULATE_CAM
    model = unet_model.module if isinstance(unet_model, nn.DataParallel) else unet_model
    model.eval()
    input_tensor = input_tensor.requires_grad_()
    output = model(input_tensor)
    target = output[:, target_index, :, :].mean()
    model.zero_grad()
    target.backward(retain_graph=True)
    activations = model.gradcam_activation
    gradients = activations.grad
    pooled_grad = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = F.relu(torch.sum(pooled_grad * activations, dim=1, keepdim=True))
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    ACCUMULATE_CAM += cam

    return ACCUMULATE_CAM


def visualize_cam(image_tensor, cam, save_path='path/to/file'):
    image = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image[..., 0:3])
    plt.title("Input Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(image[..., 0:3])
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        save_path = 'path/to/file'
    if os.path.exists(save_path):
        save_path = 'path/to/file'
    if os.path.exists(save_path):
        save_path = 'path/to/file'
    if os.path.exists(save_path):
        save_path = 'path/to/file'

    plt.savefig(save_path)
    plt.close()


# -----------------------------------------------------------------------------
# Predictor
# -----------------------------------------------------------------------------
class Predictor:


    def __init__(self, path: str, verbose: bool = True):

        assert path.exists(), f"Checkpoint {path} does not exist"

        self.path = path
        self.verbose = verbose
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.build_model()
        self.load()
        self.model.eval()
        self.to_device()

    def build_model(self):

        self.model = UNet(
            in_channels=6,
            out_channels=1,
            features=[192, 192, 192, 192],
        )

        # self.model = torch.nn.DataParallel(self.model)

    def load(self):

        with (self.path).open("rb") as f:
            state = torch.load(f, map_location=self.device)["model"]
            # self.model.load_state_dict(state["model"], strict=True)
            new_state_dict = {}
            for k, v in state.items():
                new_k = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[new_k] = v
            self.model.load_state_dict(new_state_dict, strict=True)

            if self.verbose:
                print(
                    f"Loaded checkpoint from {self.path} to {self.device}"
                )

    def to_device(self):

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.model = self.model.to(self.device)

        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameters: %.2fM" % (total / 1e6))


    def predict(self, prompts: Dict[str, any], img_features: Optional[torch.Tensor] = None,
                multimask_mode: bool = False):

        if self.verbose:
            print("point_coords", prompts.get("point_coords", None))
            print("point_labels", prompts.get("point_labels", None))
            print("box", prompts.get("box", None))
            print("img", prompts.get("img").shape, prompts.get("img").min(), prompts.get("img").max())
            if prompts.get("scribbles") is not None:
                print("scribbles", prompts.get("scribbles", None).shape, prompts.get("scribbles").min(),
                      prompts.get("scribbles").max())

        original_shape = prompts.get('img').shape[-2:]
        prompts = rescale_inputs(prompts)
        x = prepare_inputs(prompts).float()

        start_time = time.time()

        with torch.no_grad():
            # input_tensor = x.to(self.device)
            # yhat = self.model(input_tensor).cpu()
            yhat = self.model(x.to(self.device)).cpu()
            

        # input_tensor = x.to(self.device)
        # if input_tensor.requires_grad == False:
        #     input_tensor.requires_grad = True
        # yhat = self.model(input_tensor).cpu()
        # cam = compute_gradcam(self.model, input_tensor, target_index=0)
        # visualize_cam(input_tensor, cam)

        end_time = time.time()
        FPS = 1000 / ((end_time - start_time) * 1000)
        print(f"{FPS} FPS!")

        mask = torch.sigmoid(yhat)
        mask = F.interpolate(mask, size=original_shape, mode='bilinear').squeeze()

        return mask, None, yhat


# -----------------------------------------------------------------------------
# Model initialization functions
# -----------------------------------------------------------------------------

def load_model(exp_key: str = default_model):
    fpath = exp_dir / model_dict.get(exp_key)
    exp = Predictor(fpath)
    return exp, None


# -----------------------------------------------------------------------------
# Vizualization functions
# -----------------------------------------------------------------------------

def _get_overlay(img, lay, const_color="l_blue"):
    """
    Helper function for preparing overlay
    """
    assert lay.ndim == 2, "Overlay must be 2D, got shape: " + str(lay.shape)

    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)

    assert img.ndim == 3, "Image must be 3D, got shape: " + str(img.shape)

    if const_color == "blue":
        const_color = 255 * np.array([0, 0, 1])
    elif const_color == "green":
        const_color = 255 * np.array([0, 1, 0])
    elif const_color == "red":
        const_color = 255 * np.array([1, 0, 0])
    elif const_color == "l_blue":
        const_color = np.array([31, 119, 180])
    elif const_color == "orange":
        const_color = np.array([255, 127, 14])
    else:
        raise NotImplementedError

    x, y = np.nonzero(lay)
    for i in range(img.shape[-1]):
        img[x, y, i] = const_color[i]

    return img


def image_overlay(img, mask=None, scribbles=None, contour=False, alpha=0.5):
    """
    Overlay the ground truth mask and scribbles on the image if provided
    Supports both 2D grayscale and 3D RGB images.
    """
    assert img.ndim in [2, 3], "Image must be 2D (grayscale) or 3D (RGB), got shape: " + str(img.shape)

    if img.ndim == 2:
        output = np.repeat(img[..., None], 3, axis=-1)
    else:
        output = img.copy()

    if mask is not None:
        assert mask.ndim == 2, "Mask must be 2D, got shape: " + str(mask.shape)

        if contour:
            contours, _ = cv2.findContours((mask > THRESHOLD).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (0, 255, 0), 1)
        else:
            mask_overlay = _get_overlay(output, mask)
            mask2 = THRESHOLD * np.repeat(mask[..., None], 3, axis=-1)
            output = cv2.convertScaleAbs(mask_overlay * mask2 + output * (1 - mask2))

    if scribbles is not None:
        pos_scribble_overlay = _get_overlay(output, scribbles[0, ...], const_color="green")
        cv2.addWeighted(pos_scribble_overlay, alpha, output, 1 - alpha, 0, output)
        neg_scribble_overlay = _get_overlay(output, scribbles[1, ...], const_color="red")
        cv2.addWeighted(neg_scribble_overlay, alpha, output, 1 - alpha, 0, output)

    return output


def viz_pred_mask(img, mask=None, point_coords=None, point_labels=None, bbox_coords=None, seperate_scribble_masks=None,
                  binary=True):
    """
    Visualize image with clicks, scribbles, predicted mask overlaid
    """
    assert isinstance(img, np.ndarray), "Image must be numpy array, got type: " + str(type(img))
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

    if binary and mask is not None:
        mask = 1 * (mask > THRESHOLD)

    out = image_overlay(img, mask=mask, scribbles=seperate_scribble_masks)

    if point_coords is not None:
        for i, (col, row) in enumerate(point_coords):
            if point_labels[i] == 1:
                cv2.circle(out, (col, row), 2, (0, 255, 0), -1)
            else:
                cv2.circle(out, (col, row), 2, (255, 0, 0), -1)

    if bbox_coords is not None:
        for i in range(len(bbox_coords) // 2):
            cv2.rectangle(out, bbox_coords[2 * i], bbox_coords[2 * i + 1], (255, 165, 0), 1)
        if len(bbox_coords) % 2 == 1:
            cv2.circle(out, tuple(bbox_coords[-1]), 2, (255, 165, 0), -1)

    return out


# -----------------------------------------------------------------------------
# Collect scribbles
# -----------------------------------------------------------------------------

def get_scribbles(seperate_scribble_masks, last_scribble_mask, scribble_img, label: int):
    """
    Record scribbles
    """

    assert isinstance(seperate_scribble_masks, np.ndarray), \
        "seperate_scribble_masks must be numpy array, got type: " + str(type(seperate_scribble_masks))

    if scribble_img is not None:

        color_mask = scribble_img.get('mask')
        scribble_mask = color_mask[..., 0] / 255

        not_same = (scribble_mask != last_scribble_mask)
        if not isinstance(not_same, bool):
            not_same = not_same.any()

        if not_same:
            # In case any scribbles were removed
            corrected_scribble_masks = np.stack(2 * [(scribble_mask > 0)], axis=0) * seperate_scribble_masks
            corrected_last_scribble_mask = last_scribble_mask * (scribble_mask > 0)

            delta = (scribble_mask - corrected_last_scribble_mask) > 0
            new_scribbles = scribble_mask * delta
            corrected_scribble_masks[label, ...] = np.clip(corrected_scribble_masks[label, ...] + new_scribbles,
                                                           a_min=0, a_max=1)

            last_scribble_mask = scribble_mask
            seperate_scribble_masks = corrected_scribble_masks

        return seperate_scribble_masks, last_scribble_mask


def get_predictions(predictor, input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks,
                    low_res_mask, img_features, multimask_mode):
    """
    Make predictions
    """
    box = None
    if len(bbox_coords) == 1:
        gr.Error("Please click a second time to define the bounding box")
        box = None
    elif len(bbox_coords) == 2:
        box = torch.Tensor(bbox_coords).flatten()[None, None, ...].int().to(device)  # B x n x 4

    if seperate_scribble_masks is not None:
        scribble = torch.from_numpy(seperate_scribble_masks)[None, ...].to(device)
    else:
        scribble = None

    input_img = np.transpose(input_img, (2, 0, 1))

    prompts = dict(
        img=torch.from_numpy(input_img)[None, ...].to(device) / 255,
        point_coords=torch.Tensor([click_coords]).int().to(device) if len(click_coords) > 0 else None,
        point_labels=torch.Tensor([click_labels]).int().to(device) if len(click_labels) > 0 else None,
        scribbles=scribble,
        mask_input=low_res_mask.to(device) if low_res_mask is not None else None,
        box=box,
    )

    mask, img_features, low_res_mask = predictor.predict(prompts, img_features, multimask_mode=multimask_mode)

    return mask, img_features, low_res_mask


def refresh_predictions(predictor, input_img, output_img, click_coords, click_labels, bbox_coords, brush_label,
                        scribble_img, seperate_scribble_masks, last_scribble_mask,
                        best_mask, low_res_mask, img_features, binary_checkbox, multimask_mode):
    # Record any new scribbles
    seperate_scribble_masks, last_scribble_mask = get_scribbles(
        seperate_scribble_masks, last_scribble_mask, scribble_img,
        label=(0 if brush_label == "Positive (green)" else 1)  # current color of the brush
    )

    # Make prediction
    best_mask, img_features, low_res_mask = get_predictions(
        predictor, input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks, low_res_mask,
        img_features, multimask_mode
    )

    # Update input visualizations
    # mask_to_viz = best_mask.detach().numpy()
    mask_to_viz = best_mask.numpy()
    click_input_viz = viz_pred_mask(input_img, mask_to_viz, click_coords, click_labels, bbox_coords,
                                    seperate_scribble_masks, binary_checkbox)
    scribble_input_viz = viz_pred_mask(input_img, mask_to_viz, click_coords, click_labels, bbox_coords, None,
                                       binary_checkbox)

    out_viz = [
        viz_pred_mask(input_img, mask_to_viz, point_coords=None, point_labels=None, bbox_coords=None,
                      seperate_scribble_masks=None, binary=binary_checkbox),
        255 * (mask_to_viz[..., None].repeat(axis=2, repeats=3) > THRESHOLD) if binary_checkbox else mask_to_viz[
            ..., None].repeat(axis=2, repeats=3),
    ]

    return click_input_viz, scribble_input_viz, out_viz, best_mask, low_res_mask, img_features, seperate_scribble_masks, last_scribble_mask


def get_select_coords(predictor, input_img, brush_label, bbox_label, best_mask, low_res_mask,
                      click_coords, click_labels, bbox_coords,
                      seperate_scribble_masks, last_scribble_mask, scribble_img, img_features,
                      output_img, binary_checkbox, multimask_mode, autopredict_checkbox, evt: gr.SelectData):
    """
    Record user click and update the prediction
    """
    # Record click coordinates
    if bbox_label:
        bbox_coords.append(evt.index)
    elif brush_label in ['Positive (green)', 'Negative (red)']:
        click_coords.append(evt.index)
        click_labels.append(1 if brush_label == 'Positive (green)' else 0)
    else:
        raise TypeError("Invalid brush label: {brush_label}")

    # Only make new prediction if not waiting for additional bounding box click
    if (len(bbox_coords) % 2 == 0) and autopredict_checkbox:

        click_input_viz, scribble_input_viz, output_viz, best_mask, low_res_mask, img_features, seperate_scribble_masks, last_scribble_mask = refresh_predictions(
            predictor, input_img, output_img, click_coords, click_labels, bbox_coords, brush_label,
            scribble_img, seperate_scribble_masks, last_scribble_mask,
            best_mask, low_res_mask, img_features, binary_checkbox, multimask_mode
        )
        return click_input_viz, scribble_input_viz, output_viz, best_mask, low_res_mask, img_features, click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask

    else:
        click_input_viz = viz_pred_mask(
            input_img, best_mask, click_coords, click_labels, bbox_coords, seperate_scribble_masks, binary_checkbox
        )
        scribble_input_viz = viz_pred_mask(
            input_img, best_mask, click_coords, click_labels, bbox_coords, None, binary_checkbox
        )
        # Don't update output image if waiting for additional bounding box click
        return click_input_viz, scribble_input_viz, output_img, best_mask, low_res_mask, img_features, click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask


def undo_click(predictor, input_img, brush_label, bbox_label, best_mask, low_res_mask, click_coords, click_labels,
               bbox_coords,
               seperate_scribble_masks, last_scribble_mask, scribble_img, img_features,
               output_img, binary_checkbox, multimask_mode, autopredict_checkbox):
    """
    Remove last click and then update the prediction
    """
    if bbox_label:
        if len(bbox_coords) > 0:
            bbox_coords.pop()
    elif brush_label in ['Positive (green)', 'Negative (red)']:
        if len(click_coords) > 0:
            click_coords.pop()
            click_labels.pop()
    else:
        raise TypeError("Invalid brush label: {brush_label}")

    # Only make new prediction if not waiting for additional bounding box click
    if (len(bbox_coords) == 0 or len(bbox_coords) == 2) and autopredict_checkbox:

        click_input_viz, scribble_input_viz, output_viz, best_mask, low_res_mask, img_features, seperate_scribble_masks, last_scribble_mask = refresh_predictions(
            predictor, input_img, output_img, click_coords, click_labels, bbox_coords, brush_label,
            scribble_img, seperate_scribble_masks, last_scribble_mask,
            best_mask, low_res_mask, img_features, binary_checkbox, multimask_mode
        )
        return click_input_viz, scribble_input_viz, output_viz, best_mask, low_res_mask, img_features, click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask

    else:
        click_input_viz = viz_pred_mask(
            input_img, best_mask, click_coords, click_labels, bbox_coords, seperate_scribble_masks, binary_checkbox
        )
        scribble_input_viz = viz_pred_mask(
            input_img, best_mask, click_coords, click_labels, bbox_coords, None, binary_checkbox
        )

        # Don't update output image if waiting for additional bounding box click
        return click_input_viz, scribble_input_viz, output_img, best_mask, low_res_mask, img_features, click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask

    # --------------------------------------------------


with gr.Blocks(theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg)) as demo:
    # State variables
    seperate_scribble_masks = gr.State(np.zeros((2, RES, RES), dtype=np.float32))
    last_scribble_mask = gr.State(np.zeros((RES, RES), dtype=np.float32))

    click_coords = gr.State([])
    click_labels = gr.State([])
    bbox_coords = gr.State([])

    # Load default model
    predictor = gr.State(load_model()[0])
    img_features = gr.State(None)  # For SAM models
    best_mask = gr.State(None)
    low_res_mask = gr.State(None)

    gr.HTML("""\
    <h1 style="text-align: center; font-size: 28pt;">RoadGIE: Towards A Global-Scale Aerial Benchmark for Generalizable Interactive Road Extraction</h1>
    <p style="text-align: center; font-size: large;"><a href="https://anonymous.4open.science/r/RoadGIE-A2D7/">RoadGIE</a> is an interactive segmentation tool designed to help users segment road in global-scale remote sensing imagery.
    </p>

    """)

    with gr.Accordion("Open for instructions!", open=False):
        gr.Markdown(
            """
                * Select an input image from the examples below or upload your own image through the <b>'Input Image'</b> tab.
                * Use the <b>'Scribbles'</b> tab to draw <span style='color:green'>positive</span> or <span style='color:red'>negative</span> scribbles.
                    - Use the buttons in the top right hand corner of the canvas to undo or adjust the brush size
                    - Note: the app cannot detect new scribbles drawn on top of previous scribbles in a different color. Please undo/erase the scribble before drawing on the same pixel in a different color.
                * Use the <b>'Clicks/Boxes'</b> tab to draw <span style='color:green'>positive</span> or <span style='color:red'>negative</span> clicks and <span style='color:orange'>bounding boxes</span> by placing two clicks.
                * The <b>'Output'</b> tab will show the model's prediction based on your current inputs and the previous prediction.
                * The <b>'Clear Input Mask'</b> button will clear the latest prediction (which is used as an input to the model).
                * The <b>'Clear All Inputs'</b> button will clear all inputs (including scribbles, clicks, and the last prediction). 
            """
        )


    with gr.Row():
        model_dropdown = gr.Dropdown(
            label="Model",
            choices=list(model_dict.keys()),
            value=default_model,
            multiselect=False,
            interactive=False,
            visible=False
        )

    with gr.Row():
        with gr.Column(scale=1):
            brush_label = gr.Radio(["Positive (green)", "Negative (red)"],
                                   value="Positive (green)", label="Sketch/Click Label")
            bbox_label = gr.Checkbox(value=False, label="Bounding Box (2 clicks)")
        with gr.Column(scale=1):
            binary_checkbox = gr.Checkbox(value=True, label="Show binary masks", visible=False)
            autopredict_checkbox = gr.Checkbox(value=True, label="Auto-update prediction on clicks")
            gr.Markdown(
                "<span style='color:orange'>Troubleshooting:</span> If the image does not fully load in the Sketches tab, click 'Clear Sketches' or 'Clear All Inputs' to reload (it make take multiple tries). If you encounter an <span style='color:orange'>error</span> try clicking 'Clear All Inputs'.")
            multimask_mode = gr.Checkbox(value=True, label="Multi-mask mode", visible=False)

    with gr.Row():
        display_height = 500

        with gr.Column(scale=1):
            with gr.Tab("Sketches"):
                scribble_img = gr.Image(
                    label="Input",
                    brush_radius=3,
                    interactive=True,
                    brush_color="#00FF00",
                    tool="sketch",
                    height=display_height,
                    type='numpy',
                    value=default_example,
                )
                clear_scribble_button = gr.ClearButton([scribble_img], value="Clear Sketches", variant="stop")

            with gr.Tab("Clicks/Boxes") as click_tab:
                click_img = gr.Image(
                    label="Input",
                    type='numpy',
                    value=default_example,
                    height=display_height
                )
                with gr.Row():
                    undo_click_button = gr.Button("Undo Last Click")
                    clear_click_button = gr.Button("Clear Clicks/Boxes", variant="stop")

            with gr.Tab("Input Image"):
                input_img = gr.Image(
                    label="Input",
                    image_mode="RGB",  # L
                    visible=True,
                    value=default_example,
                    height=display_height
                )
                gr.Markdown(
                    "To upload your own image: click the `x` in the top right corner to clear the current image, then drag & drop")

        with gr.Column(scale=1):
            with gr.Tab("Output"):
                output_img = gr.Gallery(
                    label='Outputs',
                    columns=1,
                    elem_id="gallery",
                    preview=True,
                    object_fit="scale-down",
                    height=display_height + 50
                )

    submit_button = gr.Button("Refresh Prediction", variant='primary')
    clear_all_button = gr.ClearButton([scribble_img], value="Clear All Inputs", variant="stop")
    clear_mask_button = gr.Button("Clear Input Mask")

    # ----------------------------------------------
    # Loading Models
    # ----------------------------------------------

    model_dropdown.change(fn=load_model,
                          inputs=[model_dropdown],
                          outputs=[predictor, img_features]
                          )

    # ----------------------------------------------
    # Loading Examples
    # ----------------------------------------------

    gr.Examples(examples=test_examples,
                inputs=[input_img],
                examples_per_page=10,
                label='Examples unseen during training'
                )


    # When clear scribble button is clicked
    def clear_scribble_history(input_img):
        if input_img is not None:
            input_shape = input_img.shape[:2]
        else:
            input_shape = (RES, RES)
        return input_img, input_img, np.zeros((2,) + input_shape, dtype=np.float32), np.zeros(input_shape,
                                                                                              dtype=np.float32), None, None


    clear_scribble_button.click(clear_scribble_history,
                                inputs=[input_img],
                                outputs=[click_img, scribble_img, seperate_scribble_masks, last_scribble_mask,
                                         best_mask, low_res_mask]
                                )


    # When clear clicks button is clicked
    def clear_click_history(input_img):
        return input_img, input_img, [], [], [], None, None


    clear_click_button.click(clear_click_history,
                             inputs=[input_img],
                             outputs=[click_img, scribble_img, click_coords, click_labels, bbox_coords, best_mask,
                                      low_res_mask])


    # When clear all button is clicked
    def clear_all_history(input_img):
        if input_img is not None:
            input_shape = input_img.shape[:2]
        else:
            input_shape = (RES, RES)
        return input_img, input_img, [], [], [], [], np.zeros((2,) + input_shape, dtype=np.float32), np.zeros(
            input_shape, dtype=np.float32), None, None, None


    input_img.change(clear_all_history,
                     inputs=[input_img],
                     outputs=[click_img, scribble_img,
                              output_img, click_coords, click_labels, bbox_coords,
                              seperate_scribble_masks, last_scribble_mask,
                              best_mask, low_res_mask, img_features
                              ])

    clear_all_button.click(clear_all_history,
                           inputs=[input_img],
                           outputs=[click_img, scribble_img,
                                    output_img, click_coords, click_labels, bbox_coords,
                                    seperate_scribble_masks, last_scribble_mask,
                                    best_mask, low_res_mask, img_features
                                    ])


    # clear previous prediction mask
    def clear_best_mask(input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks):

        click_input_viz = viz_pred_mask(
            input_img, None, click_coords, click_labels, bbox_coords, seperate_scribble_masks
        )
        scribble_input_viz = viz_pred_mask(
            input_img, None, click_coords, click_labels, bbox_coords, None
        )

        return None, None, click_input_viz, scribble_input_viz


    clear_mask_button.click(
        clear_best_mask,
        inputs=[input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks],
        outputs=[best_mask, low_res_mask, click_img, scribble_img],
    )

    # ----------------------------------------------
    # Clicks
    # ----------------------------------------------

    click_img.select(get_select_coords,
                     inputs=[
                         predictor,
                         input_img, brush_label, bbox_label, best_mask, low_res_mask, click_coords, click_labels,
                         bbox_coords,
                         seperate_scribble_masks, last_scribble_mask, scribble_img, img_features,
                         output_img, binary_checkbox, multimask_mode, autopredict_checkbox
                     ],
                     outputs=[click_img, scribble_img, output_img, best_mask, low_res_mask, img_features,
                              click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask],
                     api_name="get_select_coords"
                     )

    submit_button.click(fn=refresh_predictions,
                        inputs=[
                            predictor, input_img, output_img, click_coords, click_labels, bbox_coords, brush_label,
                            scribble_img, seperate_scribble_masks, last_scribble_mask,
                            best_mask, low_res_mask, img_features, binary_checkbox, multimask_mode
                        ],
                        outputs=[click_img, scribble_img, output_img, best_mask, low_res_mask, img_features,
                                 seperate_scribble_masks, last_scribble_mask],
                        api_name="refresh_predictions"
                        )

    undo_click_button.click(fn=undo_click,
                            inputs=[
                                predictor,
                                input_img, brush_label, bbox_label, best_mask, low_res_mask, click_coords, click_labels,
                                bbox_coords,
                                seperate_scribble_masks, last_scribble_mask, scribble_img, img_features,
                                output_img, binary_checkbox, multimask_mode, autopredict_checkbox
                            ],
                            outputs=[click_img, scribble_img, output_img, best_mask, low_res_mask, img_features,
                                     click_coords, click_labels, bbox_coords, seperate_scribble_masks,
                                     last_scribble_mask],
                            api_name="undo_click"
                            )


    def update_click_img(input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks, binary_checkbox,
                         last_scribble_mask, scribble_img, brush_label, best_mask):
        """
        Draw scribbles in the click canvas
        """
        seperate_scribble_masks, last_scribble_mask = get_scribbles(
            seperate_scribble_masks, last_scribble_mask, scribble_img,
            label=(0 if brush_label == "Positive (green)" else 1)  # previous color of the brush
        )
        click_input_viz = viz_pred_mask(
            input_img, best_mask, click_coords, click_labels, bbox_coords, seperate_scribble_masks, binary_checkbox
        )
        return click_input_viz, seperate_scribble_masks, last_scribble_mask


    click_tab.select(fn=update_click_img,
                     inputs=[input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks,
                             binary_checkbox, last_scribble_mask, scribble_img, brush_label, best_mask],
                     outputs=[click_img, seperate_scribble_masks, last_scribble_mask],
                     api_name="update_click_img"
                     )


    # ----------------------------------------------
    # Scribbles
    # ----------------------------------------------

    def change_brush_color(seperate_scribble_masks, last_scribble_mask, scribble_img, label):
        """
        Recorn new scribbles when changing brush color
        """
        if label == "Negative (red)":
            brush_update = gr.Image.update(brush_color="#FF0000")  # red
        elif label == "Positive (green)":
            brush_update = gr.Image.update(brush_color="#00FF00")  # green
        else:
            raise TypeError("Invalid brush color")

        # Record latest scribbles
        seperate_scribble_masks, last_scribble_mask = get_scribbles(
            seperate_scribble_masks, last_scribble_mask, scribble_img,
            label=(1 if label == "Positive (green)" else 0)  
        )

        return seperate_scribble_masks, last_scribble_mask, brush_update


    brush_label.change(fn=change_brush_color,
                       inputs=[seperate_scribble_masks, last_scribble_mask, scribble_img, brush_label],
                       outputs=[seperate_scribble_masks, last_scribble_mask, scribble_img],
                       api_name="change_brush_color"
                       )

if __name__ == "__main__":

    demo.queue(api_open=False).launch(server_name="127.0.0.1", server_port=7860, share=True)

