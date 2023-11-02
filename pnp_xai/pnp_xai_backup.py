import os
import warnings
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
# from captum.attr import visualization as viz

from pnp_xai.detector.detector import ModelArchitectureDetectorV2
from pnp_xai.recommender.recommender import XaiRecommender
from pnp_xai.evaluator.evaluator import XaiEvaluator
# from pnp_xai.explainers import get_configs
from pnp_xai.explainers.config import attribute_kwargs
from utils import class_to_string

import pdb


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams["font.family"] = "Times New Roman"


class PnPXAI:
    def __init__(self):
        self.detector = ModelArchitectureDetectorV2()
        self.recommender = XaiRecommender()
        self.evaluator = XaiEvaluator()

        # self.configs = get_configs()

    def _postprocess_attr(self, attr, sign='absolute'):
        if sign == 'absolute':
            attr = torch.abs(attr)
        elif sign == 'positive':
            attr = torch.nn.functional.relu(attr)
        else:
            raise NotImplementedError

        postprocessed = attr.permute((1, 2, 0)).sum(dim=-1)
        attr_max = torch.max(postprocessed)
        attr_min = torch.min(postprocessed)
        postprocessed = (postprocessed - attr_min) / (attr_max - attr_min)
        return postprocessed.cpu().detach().numpy()

    def explain(self, user_input, sample_idx=966, savedir='./results'):
        if isinstance(sample_idx, int):
            sample_idx = [sample_idx]

        self.device = next(user_input.model.parameters()).device
        detector_output = self.detector(
            model=user_input.model,
            sample=user_input.data.dataset[0][0][None, :].to(self.device)
        )

        recommender_output = self.recommender(
            question=user_input.question,
            task=user_input.task,
            architecture=detector_output.architecture,
        )

        # evaluator_output = self.evaluator(
        #     model=user_input.model,
        #     sample=user_input.data.dataset[idx][0][None, :].to(self.device),
        #     explainers=recommender_output.explainers,
        #     evaluation_metrics=recommender_output.evaluation_metrics,
        # )

        # Initialize explainers
        explainers = []
        for explainer in recommender_output.explainers:
            try:
                explainers.append(explainer(user_input.model))
            except NotImplementedError as e:
                warnings.warn(f"\n[Explainer] Warning: {explainer} is not currently supported.")
            except TypeError as e:
                if explainer.__name__ == 'GuidedGradCam':
                    if class_to_string(user_input.model) == 'VGG':
                        explainers.append(
                            explainer(user_input.model, layer=user_input.model.features[24])
                        )
                    elif class_to_string(user_input.model) == 'ResNet':
                        explainers.append(
                            explainer(user_input.model, layer=user_input.model.layer1[0].conv2)
                        )
                    elif class_to_string(user_input.model) == 'ViT':
                        pass
                else:
                    warnings.warn(f"\n[Explainer] Warning: {explainer} is not currently supported.")

        # pdb.set_trace()
        for i, idx in enumerate(sample_idx):
            sample = user_input.data.dataset[idx][0][None, :].to(self.device)
            pred_score, pred_idx = user_input.model(sample).topk(1)

            # pdb.set_trace()

            # Visualization Ver. 1
            ncols = len(explainers)+1
            fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4*(ncols), 4), gridspec_kw={'width_ratios': [1]*ncols})

            axs[0].set_title('sample')
            axs[0].imshow(np.transpose(sample[0].cpu().detach().numpy(), (1, 2, 0)))
            
            results = {}
            for j, explainer in enumerate(explainers):
                method = class_to_string(explainer)
                results[method] = explainer.attribute(
                    inputs=sample,
                    target=pred_idx,
                    **attribute_kwargs[method],
                )
                # pdb.set_trace()

                axs[j+1].set_title(method)
                im = axs[j+1].imshow(self._postprocess_attr(results[method][0]), cmap='gist_heat')

            for ax in axs:
                ax.axis('off')

            cax = fig.add_axes([axs[-1].get_position().x1+0.01, axs[-1].get_position().y0, 0.015, axs[-1].get_position().height])
            plt.colorbar(im, cax=cax)

            # savedir = './results'
            os.makedirs(savedir, exist_ok=True)
            plt.savefig(os.path.join(savedir, f'{idx}.png'))


            # Visualization Ver. 2
            ncols = len(explainers)
            fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4*(ncols), 4), gridspec_kw={'width_ratios': [1]*ncols})

            for j, explainer in enumerate(explainers):
                method = class_to_string(explainer)

                axs[j].set_title(method)
                axs[j].imshow(np.transpose(sample[0].cpu().detach().numpy(), (1, 2, 0)))
                im = axs[j].imshow(self._postprocess_attr(results[method][0]), alpha=0.8, cmap='gist_heat')

            for ax in axs:
                ax.axis('off')

            cax = fig.add_axes([axs[-1].get_position().x1+0.01, axs[-1].get_position().y0, 0.015, axs[-1].get_position().height])
            plt.colorbar(im, cax=cax)

            # savedir = './results'
            os.makedirs(savedir, exist_ok=True)
            plt.savefig(os.path.join(savedir, f'{idx}_overlaid.png'))