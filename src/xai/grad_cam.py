import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    HAS_CV2 = False

class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for Brain Tumor MRI Segmentation Model Interpretability.
    Visualizes features driving segmentation predictions for specified tumor classes.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=3):
        """
        input_tensor: (1, 4, H, W)
        target_class: tumor class index to generate interpretability heatmap for (e.g. 3 = Enhancing Tumor)
        Returns: normalized heatmap (H, W) in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor) # (1, C, H, W)
        
        # Target score: sum of activations for target class region
        score = torch.sum(output[0, target_class])
        score.backward(retain_graph=True)

        gradients = self.gradients[0].cpu().data.numpy()   # (C, H, W)
        activations = self.activations[0].cpu().data.numpy() # (C, H, W)

        # Grad-CAM++ weight calculations
        grad_sq = gradients ** 2
        grad_cube = grad_sq * gradients
        
        sum_activations = np.sum(activations, axis=(1, 2), keepdims=True)
        aij = grad_sq / (2 * grad_sq + sum_activations * grad_cube + 1e-7)
        
        weights = np.sum(aij * np.maximum(gradients, 0), axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        if HAS_CV2 and cv2 is not None:
            cam_resized = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        else:
            cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
            cam_resized_tensor = F.interpolate(cam_tensor, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
            cam_resized = cam_resized_tensor.squeeze(0).squeeze(0).numpy()

        return cam_resized
