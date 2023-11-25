from os import makedirs
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.logger import Logger
from src.model.detector import Detector
from src.model.loss import YoloLoss
from src.model.yolo import Yolo
from src.utils import (
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    non_max_suppression,
    compute_metrics,
)


class Trainer:
    def __init__(
            self,
            config: Dict[str, Any],
            dataloaders: Tuple[DataLoader, DataLoader, DataLoader],
            device: torch.device,
            save_dir: Path,
            checkpoint_path: Optional[Path]
    ) -> None:
        self.logger = Logger()
        self.device = device

        learning_rate = config["learning_rate"]
        self.iou_threshold = config["iou_threshold"]
        self.conf_threshold = config["conf_threshold"]
        self.epochs = config["epochs"]
        self.eval_interval = config["eval_interval"]
        self.checkpoint_interval = config["checkpoint_interval"]
        self.max_grad_norm = config["max_grad_norm"]

        self.model = Yolo(config).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = YoloLoss(config)

        self.save_dir = save_dir / self.logger.tensorboard_dir.name
        makedirs(self.save_dir, exist_ok=True)

        if checkpoint_path is not None:
            self.model = load_checkpoint(checkpoint_path, self.model)
            self.logger.info("Model loader from checkpoint")

        self.logger.info(f"Number of trainable parameters: {count_parameters(self.model)}")

        self.train_dl, self.val_dl, self.test_dl = dataloaders
        self.best_score_metric = "F1"
        self.best_score = 0

        self.detector = Detector(config, self.model)

    def fit(self) -> Module:
        for epoch in range(1, self.epochs + 1):
            self.logger.info(f"[Epoch {epoch} / {self.epochs}]")
            self._train(epoch)

            if epoch % self.eval_interval == 0:
                self.evaluate(self.val_dl, self.iou_threshold, self.conf_threshold, epoch)

        self.logger.close()
        return self.model

    def _train(self, epoch: int) -> None:
        self.model.train()

        for batch in tqdm(self.train_dl):
            loss, _ = self._forward(batch)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

        self.logger.log_epoch(mode="train", epoch=epoch)

    def evaluate(
            self,
            dataloader: DataLoader,
            iou_threshold: float,
            threshold: float,
            epoch: Optional[int] = None,
    ) -> None:
        self.model.eval()

        all_predicted_boxes = []
        all_target_boxes = []
        entry_idx = 0

        for batch in tqdm(dataloader):
            self.optimizer.zero_grad()

            source = batch[0].to(self.device)
            target = batch[1].to(self.device)

            with torch.no_grad():
                _, output = self._forward(batch)

            target_bboxes = self.detector.cellboxes_to_boxes(target)
            predicted_bboxes = self.detector.cellboxes_to_boxes(output)

            batch_size = source.shape[0]
            for idx in range(batch_size):
                nms_boxes = non_max_suppression(
                    predicted_bboxes[idx],
                    iou_threshold=iou_threshold,
                    threshold=threshold,
                )

                for nms_box in nms_boxes:
                    all_predicted_boxes.append([entry_idx] + nms_box)

                for box in target_bboxes[idx]:
                    if box[0] > threshold:
                        all_target_boxes.append([entry_idx] + box)

                entry_idx += 1

        metrics = compute_metrics(all_predicted_boxes, all_target_boxes, iou_threshold)
        self.logger.log_epoch(mode="eval", epoch=epoch, metrics=metrics)

        if epoch is not None:
            if self.best_score < metrics[self.best_score_metric]:
                self.best_score = metrics[self.best_score_metric]
                self.logger.info(f"Saving best model according to {self.best_score_metric}: {self.best_score:.3f}")
                save_checkpoint(self.save_dir / f"checkpoint_best_{self.best_score_metric.lower()}.pth", self.model)

            if epoch % self.checkpoint_interval == 0:
                self.logger.info(f"Saving checkpoint at epoch: {epoch}")
                save_checkpoint(self.save_dir / f"checkpoint_{epoch}.pth", self.model)

    def _forward(self, batch: Tensor
                 ) -> Tuple[Tensor, Tensor]:
        self.optimizer.zero_grad()

        source = batch[0].to(self.device)
        target = batch[1].to(self.device)

        output = self.model(source)
        losses = self.criterion(output, target)

        self.logger.log_losses(losses)
        loss, *_ = losses
        return loss, output
