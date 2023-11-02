from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
import torch

from pnp_xai.explainers.config import attribute_kwargs
from utils import class_to_string
from pnp_xai.evaluator.config import infidelity_kwargs, sensitivity_kwargs

import pdb
import time

# @dataclass
# class EvaluatorOutput:
#     explainers: list
#     evaluation_results: list
@dataclass
class EvaluatorOutput:
    explanation_results: dict
    evaluation_results: dict
    infidelity_results: dict
    sensitivity_results: dict


class XaiEvaluator:
    def __init__(self):
        pass

    def infidelity(self, model, sample, label, pred, result, explainer):
        repeated_sample = sample.repeat(infidelity_kwargs['num_repeat'], 1, 1, 1)

        # Add Gaussian random noise
        std, mean = torch.std_mean(repeated_sample)
        noise = torch.randn(size=repeated_sample.shape).to(self.device) * std + mean
        perturbed_sample = infidelity_kwargs['noise_scale'] * noise + repeated_sample
        perturbed_sample = torch.minimum(repeated_sample, perturbed_sample)
        perturbed_sample = torch.maximum(repeated_sample-1, perturbed_sample)

        # Compute the dot product of the input perturbation to the explanation
        dot_product = torch.mul(perturbed_sample, result)
        dot_product = dot_product.sum(dim=(1, 2, 3))
        mu = torch.ones(dot_product.shape).to(self.device)

        def _forward_batch(model, samples, label, batch_size):
            training_mode = model.training
            model.eval()

            predictions = torch.zeros(samples.shape[0]).to(self.device)
            cur_idx = 0
            next_idx = min(batch_size, samples.shape[0])
            while cur_idx < samples.shape[0]:
                idxs = np.arange(cur_idx, next_idx)
                predictions[idxs] = model(samples[idxs])[:, label].detach()

                cur_idx = next_idx
                next_idx += batch_size
                next_idx = min(next_idx, samples.shape[0])

            model.training = training_mode
            return predictions

        # Compute the difference between the original prediction and the perturbed prediction
        # TODO: Check model(repeated_sample) vs. model(perturbed_sample)
        # perturbed_pred = _forward_batch(model, repeated_sample, label, infidelity_kwargs['batch_size'])
        perturbed_pred = _forward_batch(model, perturbed_sample, label, infidelity_kwargs['batch_size'])
        pred_diff = pred - perturbed_pred

        ''' Notes
        - `pred_diff` is extremely small; close to zero.
        - It makes `scaling_factor` to almost zero.
        - `pred_diff - dot_product` is also almost zero.
        - 
        '''

        scaling_factor = torch.mean(mu * pred_diff * dot_product) / torch.mean(mu * dot_product * dot_product)
        dot_product *= scaling_factor
        infd = torch.mean(mu * torch.square(pred_diff - dot_product)) / torch.mean(mu)

        return infd

    def sensitivity(self, sample, result, explainer, pred_idx):
        method = class_to_string(explainer)
        norm = torch.linalg.norm(result)

        sens = torch.tensor(-torch.inf)
        epsilon = sensitivity_kwargs['epsilon']
        for _ in range(sensitivity_kwargs['num_iter']):
            # Add random uniform noise which ranges [-epsilon, epsilon]
            noise = torch.rand(size=sample.shape).to(self.device)
            noise = noise * epsilon * 2 - epsilon
            perturbed_sample = sample + noise

            # Get perturbed explanation results
            perturbed_result = explainer.attribute(
                inputs=perturbed_sample,
                target=pred_idx,
                **attribute_kwargs[method],
            )

            # Get maximum of the difference between the perturbed explanation and the original explanation
            sens = torch.max(sens, torch.linalg.norm(result - perturbed_result)/norm)
        return sens

    def prioritize(self, metrics, weights):
        assert sum(weights) == 1, "Sum of weights should be 1."
        num_metric = len(weights)
        weighted_scores = dict()
        for method in metrics[0].keys():
            weighted_score = 0
            for i, weight in enumerate(weights):
                weighted_score += metrics[i][method] * weight
            weighted_scores[method] = weighted_score

        weighted_scores = OrderedDict(sorted(weighted_scores.items(), key=lambda item: item[1]))
        return weighted_scores

    def __call__(self, model, sample, label, explainers, evaluation_metrics):
        self.device = sample.device
        pred = model(sample)
        pred_score, pred_idx = pred.topk(1)
        
        results = dict()
        infd_scores = dict()
        sens_scores = dict()

        for explainer in explainers:
            method = class_to_string(explainer)
            print(method)

            # Get attribution score
            st = time.time()
            results[method] = explainer.attribute(
                inputs=sample,
                target=pred_idx,
                **attribute_kwargs[method],
            )
            print(f'Compute attribution done: {time.time() - st}')
            st = time.time()

            # Compute infidelity score
            infd_scores[method] = self.infidelity(model, sample, label, pred[:, label], results[method], explainer)
            print(f'Compute infidelity done: {time.time() - st}')
            st = time.time()

            # Compute sensitivity score
            sens_scores[method] = self.sensitivity(sample, results[method], explainer, pred_idx)
            print(f'Compute sensitivity done: {time.time() - st}\n')

        # Prioritize explanation results by weighted score
        weighted_scores = self.prioritize(
            metrics=[infd_scores, sens_scores],
            weights=[0.5, 0.5]
        )

        # return EvaluatorOutput(
        #     explainers=...,
        #     evaluation_results=...,
        # )
        return EvaluatorOutput(
            explanation_results=results,
            evaluation_results=weighted_scores,
            infidelity_results=infd_scores,
            sensitivity_results=sens_scores,
        )
