import base64
import io
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .preprocess import preprocess_pil


class GradCAM:
    """Grad-CAM implementation for ResNet models."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap via forward+backward pass."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def get_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "layer4"):
        return model.layer4
    raise ValueError("Cannot find target layer for Grad-CAM")


def generate_heatmap_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay Grad-CAM heatmap on the original image."""
    heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    original_np = np.array(original_image)
    overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(overlay)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to PNG base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def generate_gradcam_base64(
    model: nn.Module,
    image: Image.Image,
    device: Optional[str] = None,
    target_class: Optional[int] = None,
) -> str:
    """Generate Grad-CAM heatmap and return as base64 PNG."""
    if device is None:
        device = next(model.parameters()).device

    input_tensor = preprocess_pil(image).to(device)
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    try:
        input_tensor.requires_grad_(True)
        heatmap = gradcam.generate(input_tensor, target_class=target_class)
    finally:
        gradcam.remove_hooks()

    overlay_image = generate_heatmap_overlay(image, heatmap)
    return image_to_base64(overlay_image)
