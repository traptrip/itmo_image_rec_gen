from typing import Any
import torch
import torch.nn as nn
from ..utils import load_obj  # noqa
from .base_trainer import Trainer  # noqa


class TorchVisionTrainer(Trainer):
    def _initialize_model(self) -> Any:
        model_cfg = self.cfg.model
        model = load_obj(model_cfg._target_)(**model_cfg.params)
        if model_cfg.freeze:
            for mp in model.parameters():
                mp.requires_grad = False
        if "resnet" in model_cfg._target_.lower():
            model.fc = nn.Linear(model.fc.in_features, model_cfg.n_classes)
        elif "mobilenet" in model_cfg._target_.lower():
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, model_cfg.n_classes
            )
        elif "vit" in model_cfg._target_.lower():
            model.heads.head = torch.nn.Linear(
                model.heads.head.in_features, model_cfg.n_classes
            )
        else:
            raise ValueError("No such model!")
        if self.cfg.general.pretrained_weights:
            weights = torch.load(self.cfg.general.pretrained_weights)
            model.load_state_dict(weights)
        return model
