import os
import warnings
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from pnp_xai.detector.detector import ModelArchitectureDetectorV2
from pnp_xai.recommender.recommender import XaiRecommender
from pnp_xai.evaluator.evaluator import XaiEvaluator
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

    def _postprocess_attr(self, attr, sign='absolute'):
        if sign == 'absolute':
            attr = torch.abs(attr)
        elif sign == 'positive':
            attr = torch.nn.functional.relu(attr)
        elif sign == 'negative':
            attr = -torch.nn.functional.relu(-attr)
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

        # Initialize explainers
        explainers = []
        for explainer in recommender_output.explainers:
            # pdb.set_trace()
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

        for i, idx in enumerate(sample_idx):
            sample = user_input.data.dataset[idx][0][None, :].to(self.device)
            label = user_input.data.dataset[idx][1]

            evaluator_output = self.evaluator(
                model=user_input.model,
                sample=sample,
                label=label,
                explainers=explainers,
                evaluation_metrics=recommender_output.evaluation_metrics,
            )
            results = evaluator_output.explanation_results
            total_scores = evaluator_output.evaluation_results
            infd_scores = evaluator_output.infidelity_results
            sens_scores = evaluator_output.sensitivity_results

            # pdb.set_trace()

            # Visualize explanations
            ncols = len(explainers)+1
            fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4*(ncols), 4), gridspec_kw={'width_ratios': [1]*ncols})

            axs[0].set_title('sample', fontsize=18)
            axs[0].imshow(np.transpose(sample[0].cpu().detach().numpy(), (1, 2, 0)))

            for j, method in enumerate(total_scores.keys()):
                axs[j+1].set_title(f'{method}', fontsize=18)
                axs[j+1].set_xlabel(
                    f'weighted score: {total_scores[method].item():.4f}\n\
                    infidelity: {infd_scores[method].item():.4f}\n\
                    sensitivity: {sens_scores[method].item():.4f}\n',
                    fontsize=15,
                )
                # im = axs[j+1].imshow(self._postprocess_attr(results[method][0], sign='absolute'), cmap='gist_heat')
                im = axs[j+1].imshow(self._postprocess_attr(results[method][0], sign='positive'), cmap='gist_heat')

            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])

            cax = fig.add_axes([axs[-1].get_position().x1+0.01, axs[-1].get_position().y0, 0.015, axs[-1].get_position().height])
            plt.colorbar(im, cax=cax)

            os.makedirs(savedir, exist_ok=True)
            plt.savefig(os.path.join(savedir, f'{idx}.png'))
