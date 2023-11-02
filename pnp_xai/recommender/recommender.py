from dataclasses import dataclass
import warnings
import torch.nn as nn

from pnp_xai.explainers import *
# from pnp_xai.evaluator import correctness, continuity

import pdb
# pdb.set_trace()


@dataclass
class RecommenderOutput:
    explainers: list
    evaluation_metrics: list


class XaiRecommender:
    def __init__(self):
        self.question_table = {
            'why': [GuidedGradCam, Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, TCAV, Anchors],
            'how': [PDP],
            'why not': [CEM],
            'how to still be this': [Anchors],
        }
        self.task_table = {
            'image': [GuidedGradCam, Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV],
            'tabular': [Lime, KernelShap, PDP, CEM, Anchors],
            'text': [Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM],
        }
        self.architecture_table = {
            nn.Linear: [Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV, Anchors],
            nn.Conv1d: [GuidedGradCam, Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV, Anchors],
            nn.Conv2d: [GuidedGradCam, Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV, Anchors],
            nn.RNN: [Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV, Anchors],
            nn.Transformer: [Lime, KernelShap, IntegratedGradients, FullGrad, CEM, TCAV, Anchors],
            nn.MultiheadAttention: [Lime, KernelShap, IntegratedGradients, FullGrad, CEM, TCAV, Anchors],
        }
        self.evaluation_metric_table = {
            GuidedGradCam: ['Correctness', 'Coherence'],
            Lime: ['Correctness'],
            KernelShap: ['Correctness'],
            IntegratedGradients: ['Correctness', 'Coherence'],
            FullGrad: ['Correctness', 'Coherence'],
            LRP: ['Correctness', 'Coherence'],
            RAP: ['Correctness', 'Coherence'],

            # Evaluation metric not implemented yet
            PDP: [None],
            CEM: ['Completeness'],
            TCAV: ['Correctness'],
            Anchors: ['Completeness', 'Compactness'],
        }

    def _find_overlap(self, lists):
        if not lists:
            return []
        result = set(lists[0])
        for lst in lists[1:]:
            result = result.intersection(lst)
        return list(result)

    def filter_methods(self, question, task, architecture):
        question_to_method = self.question_table[question]
        task_to_method = self.task_table[task]
        
        architecture_to_method = []
        for module in architecture:
            try:
                architecture_to_method.append(self.architecture_table[module])
            except KeyError:
                warnings.warn(f"\n[Recommender] Warning: {repr(module)} is not currently supported.")
        architecture_to_method = self._find_overlap(architecture_to_method)
        if (nn.Conv1d in architecture or nn.Conv2d in architecture) and GuidedGradCam not in architecture_to_method:
            if nn.MultiheadAttention not in architecture:
                architecture_to_method.append(GuidedGradCam)
        
        methods = self._find_overlap([question_to_method, task_to_method, architecture_to_method])
        
        return methods

    def suggest_metrics(self, methods):
        method_to_metric = []
        for method in methods:
            method_to_metric.append(self.evaluation_metric_table[method])
        metrics = self._find_overlap(method_to_metric)
        return metrics

    def __call__(self, question, task, architecture):
        methods = self.filter_methods(question, task, architecture)
        metrics = self.suggest_metrics(methods)
        return RecommenderOutput(
            explainers=methods,
            evaluation_metrics=metrics,
        )