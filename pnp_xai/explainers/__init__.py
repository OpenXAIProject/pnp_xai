import yaml
import pdb

# Captum implementation
from pnp_xai.explainers.integrated_gradients import IntegratedGradients
from pnp_xai.explainers.lrp import LRP
from pnp_xai.explainers.guided_gradcam import GuidedGradCam
from pnp_xai.explainers.kernel_shap import KernelShap
from pnp_xai.explainers.lime import Lime


# Custom implementation
# TODO: Implement custom LRP to handle residual connections and LayerNorm.

# Not supported yet
from pnp_xai.explainers.fullgrad import FullGrad
from pnp_xai.explainers.rap import RAP
from pnp_xai.explainers.tcav import TCAV
from pnp_xai.explainers.anchors import Anchors
from pnp_xai.explainers.cem import CEM
from pnp_xai.explainers.pdp import PDP


def get_configs():
    with open('./pnp_xai/explainers/config.yml') as f:
        configs = yaml.full_load(f)
    # configs = configs[explainer]
    # pdb.set_trace()
    return configs