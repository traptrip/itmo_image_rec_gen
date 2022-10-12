import json
import pathlib
from typing import Any, Tuple

import cv2
from torch.utils.data import Dataset
from omegaconf import DictConfig

from ..utils import load_augs


class FruitsDataset(Dataset):
    """<https://www.kaggle.com/datasets/moltean/fruits>"""

    def __init__(self, cfg: DictConfig, stage: str = "train") -> None:
        """
        Args:
            cfg (DictConfig): Config.
            stage (string): The dataset split, supports ``"train"`` (default), or ``"val"``.
        """
        super().__init__()

        data_folder = pathlib.Path(cfg.data.root) / stage
        label2id = {
            img_p.parts[-1]: i for i, img_p in enumerate(sorted(data_folder.iterdir()))
        }
        with open("label2id.txt", "w") as f:
            json.dump(label2id, f)
        self._samples = [
            [
                cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB),
                label2id[label.name],
            ]
            for label in data_folder.iterdir() for img_p in label.iterdir()
        ]
        self.transform = (
            load_augs(cfg.transform.train.augs)
            if stage == "train"
            else load_augs(cfg.transform.valid.augs)
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image, target = self._samples[idx]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, target
