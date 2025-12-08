# model/gradcam.py

from typing import Optional, Tuple
import base64
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .preprocess import preprocess_pil


class GradCAM:
    """
    Grad-CAM implementation for ResNet models.
    Extracts activation maps from the last convolutional layer.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W)
            target_class: Target class index. If None, use the predicted class.
        
        Returns:
            Heatmap as numpy array (H, W) with values in [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Apply ReLU
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Get the target layer for Grad-CAM based on model architecture.
    For ResNet models, this is layer4.
    """
    if hasattr(model, 'layer4'):
        return model.layer4
    raise ValueError("Cannot find target layer for Grad-CAM")


def generate_heatmap_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> Image.Image:
    """
    Overlay heatmap on the original image.
    
    Args:
        original_image: Original PIL Image
        heatmap: Grad-CAM heatmap (H, W) with values in [0, 1]
        alpha: Transparency of the heatmap overlay
    
    Returns:
        PIL Image with heatmap overlay
    """
    import cv2
    
    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))
    
    # Apply colormap (jet)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Convert original image to numpy
    original_np = np.array(original_image)
    
    # Blend
    overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(overlay)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def generate_gradcam_base64(
    model: nn.Module,
    image: Image.Image,
    device: Optional[str] = None
) -> Tuple[str, str]:
    """
    Generate Grad-CAM heatmap and return as base64.
    
    Args:
        model: PyTorch model
        image: PIL Image
        device: Device to run inference on
    
    Returns:
        Tuple of (label, heatmap_base64)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Preprocess
    input_tensor = preprocess_pil(image).to(device)
    
    # Get target layer
    target_layer = get_target_layer(model)
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Forward pass to get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        # target_class = 1  # FAKE class for visualization - or use predicted class
        target_class = output.argmax(dim=1).item()
    
    # Enable gradients for Grad-CAM
    input_tensor.requires_grad_(True)
    heatmap = gradcam.generate(input_tensor, target_class=target_class)
    
    # Create overlay
    overlay_image = generate_heatmap_overlay(image, heatmap)
    
    # Convert to base64
    heatmap_base64 = image_to_base64(overlay_image)
    
    return heatmap_base64
