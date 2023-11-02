import captum
import torch
from skimage.segmentation import felzenszwalb

class KernelShap:
    def __init__(self, model):
        self.model = model
        self.kernel_shap = captum.attr.KernelShap(self.model)
        self.device = next(self.model.parameters()).device

    def attribute(self, inputs, target, n_samples):
        inputs_np = inputs.permute(0, 2, 3, 1).cpu().numpy()
        seg = felzenszwalb(inputs_np[0], scale=250)
        seg = torch.tensor(seg).cuda().to(self.device)
        
        attribution = self.kernel_shap.attribute(
            inputs, 
            target=target,
            n_samples=n_samples, 
            feature_mask=seg,
            perturbations_per_eval=16
        )
        
        return attribution
