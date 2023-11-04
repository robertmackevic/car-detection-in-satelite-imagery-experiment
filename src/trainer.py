import torch
import logging

from collections import Counter
from os import makedirs, listdir
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from tqdm import tqdm
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from src.model.yolo import Yolo
from src.model.loss import YoloLoss
from src.utils import (
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    non_max_suppression,
    intersection_over_union
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
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)
        self.device = device

        self.model = Yolo(config).to(device)

        learning_rate = config["learning_rate"]
        self.num_splits = config["splits"]
        self.num_boxes = config["boxes"]
        self.num_classes = config["classes"]
        self.iou_threshold = config["iou_threshold"]
        self.conf_threshold = config["conf_threshold"]
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        self.criterion = YoloLoss(config)

        self.save_dir = save_dir
        self.tensorboard_dir = Path(".runs")
        makedirs(self.save_dir, exist_ok=True)
        makedirs(self.tensorboard_dir, exist_ok=True)

        self.epochs = config["epochs"]
        self.eval_interval = config["eval_interval"]
        self.checkpoint_interval = config["checkpoint_interval"]

        if checkpoint_path is not None:
            self.model = load_checkpoint(checkpoint_path, self.model)
            self.logger.info("Model loader from checkpoint")
        self.logger.info(f"Number of trainable parameters: {count_parameters(self.model)}")
        self.tensorboard = SummaryWriter(self.tensorboard_dir / f"v{len(listdir(self.tensorboard_dir))}")

        self.train_dl, self.val_dl, self.test_dl = dataloaders
        self.best_score = 0

        self.logs = {
            "loss": [],
            "box_loss": [],
            "object_loss": [],
            "no_object_loss": [],
            "class_loss": [],
        }

    def fit(self) -> Module:
        for epoch in range(1, self.epochs + 1):
            self._clear_logs()
            self.logger.info(f"[Epoch {epoch} / {self.epochs}]")
            self._train(epoch)

            if epoch % self.eval_interval == 0:
                self.evaluate(self.val_dl, epoch)

        self.tensorboard.close()

    def _train(self, epoch: int) -> None:
        self.model.train()

        for batch in tqdm(self.train_dl):
            losses, _ = self._forward(batch)
            loss, *_ = losses
            loss.backward()
            self.optimizer.step()

        self._log_epoch(epoch, mode="train")

    def evaluate(self, dataloader: DataLoader, epoch: Optional[int] = None) -> None:
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
            
            target_bboxes = self.cellboxes_to_boxes(target)
            predicted_bboxes = self.cellboxes_to_boxes(output)

            batch_size = source.shape[0]

            for idx in range(batch_size):
                nms_boxes = non_max_suppression(
                    predicted_bboxes[idx],
                    iou_threshold=self.iou_threshold,
                    threshold=self.conf_threshold,
                )

                for nms_box in nms_boxes:
                    all_predicted_boxes.append([entry_idx] + nms_box)

                for box in target_bboxes[idx]:
                    if box[1] > self.conf_threshold:
                        all_target_boxes.append([entry_idx] + box)

                entry_idx += 1

        mean_ap = self.mean_average_precision(all_predicted_boxes, all_target_boxes)
        self.logger.info(f"mAP: {mean_ap}")

        if epoch is not None:
            if self.best_score < mean_ap:
                save_checkpoint(self.save_dir / "checkpoint_best_map.pth", self.model)
                self.best_score = mean_ap
        
            if epoch % self.checkpoint_interval == 0:
                save_checkpoint(self.save_dir / f"checkpoint_{epoch}.pth", self.model)

            self._log_epoch(epoch, mode="eval", mean_ap=mean_ap)

    def _forward(self, batch: Tensor
        ) -> Tuple[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]:
        self.optimizer.zero_grad()

        source = batch[0].to(self.device)
        target = batch[1].to(self.device)

        output = self.model(source)
        losses = self.criterion(output, target)
        
        self._log_losses(losses)        
        return losses, output

    def _log_losses(self, losses: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> None:
        loss, box_loss, object_loss, no_object_loss, class_loss = losses
        self.logs["loss"].append(loss.item())
        self.logs["box_loss"].append(box_loss.item())
        self.logs["object_loss"].append(object_loss.item())
        self.logs["no_object_loss"].append(no_object_loss.item())
        self.logs["class_loss"].append(class_loss.item())

    def _clear_logs(self) -> None:
        self.logs = {key: [] for key, _ in self.logs.items()}

    def _log_epoch(self, epoch: int, mode: str, mean_ap: Optional[float] = None) -> None:
        message = ""
        for metric, values in self.logs.items():
            mean = sum(values) / len(values)
            message += f"{metric}: {mean:.3f} "
            self.tensorboard.add_scalar(f"{metric}/{mode}", mean, epoch)
        
        self.logger.info(message)

        if mean_ap is not None:
            self.tensorboard.add_scalar(f"mAP", mean_ap, epoch)

    def cellboxes_to_boxes(self, cells: Tensor) -> List[List[float]]:
        converted_pred = self.convert_cellboxes(cells).reshape(cells.shape[0], self.num_splits * self.num_splits, -1)
        converted_pred[..., 0] = converted_pred[..., 0].long()
        all_bboxes = []

        for ex_idx in range(cells.shape[0]):
            bboxes = []

            for bbox_idx in range(self.num_splits * self.num_splits):
                bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
            all_bboxes.append(bboxes)

        return all_bboxes
    
    def convert_cellboxes(self, cells: Tensor) -> Tensor:
        cells = cells.to("cpu")
        batch_size = cells.shape[0]

        cells = cells.reshape(batch_size, self.num_splits, self.num_splits, self.num_classes + self.num_boxes * 5)
        bboxes = [cells[..., self.num_classes + 1 + (5 * i): self.num_classes + 5 + (5 * i)] for i in range(self.num_boxes)]

        scores = torch.cat([
            cells[..., self.num_classes + (5 * i)].unsqueeze(0)
            for i in range(self.num_boxes)
        ],dim=0)

        best_box = scores.argmax(0).unsqueeze(-1)
        best_boxes = torch.zeros_like(bboxes[0]) 
        for i in range(self.num_boxes):
            best_boxes += best_box.eq(i).float() * bboxes[i]

        cell_indices = torch.arange(self.num_splits).repeat(batch_size, self.num_splits, 1).unsqueeze(-1)

        x = 1 / self.num_splits * (best_boxes[..., :1] + cell_indices)
        y = 1 / self.num_splits * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
        w_y = 1 / self.num_splits * best_boxes[..., 2:4]

        converted_bboxes = torch.cat((x, y, w_y), dim=-1)

        predicted_class = cells[..., :self.num_classes].argmax(-1).unsqueeze(-1)

        best_confidence = torch.max(
            torch.stack([
                cells[..., self.num_classes + (5 * i)] for i in range(self.num_boxes)
            ], dim=0),dim=0).values.unsqueeze(-1)

        converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)
        return converted_preds
    
    def mean_average_precision(self, pred_boxes: List, true_boxes: List) -> float:
        epsilon = 1e-6
        aps = []

        for c in range(self.num_classes):
            detections = []
            ground_truths = []

            for detection in pred_boxes:
                if detection[1] == c:
                    detections.append(detection)

            for true_box in true_boxes:
                if true_box[1] == c:
                    ground_truths.append(true_box)

            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            detections.sort(key=lambda x: x[2], reverse=True)
            tp = torch.zeros((len(detections)))
            fp = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)
            
            # If none exists for this class then we can safely skip
            if total_true_bboxes == 0:
                continue

            for detection_idx, detection in enumerate(detections):
                ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[0] == detection[0]
                ]

                best_iou = 0

                for idx, gt in enumerate(ground_truth_img):
                    iou = intersection_over_union(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > self.iou_threshold:
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        tp[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        fp[detection_idx] = 1

                else:
                    fp[detection_idx] = 1

            TP_cumsum = torch.cumsum(tp, dim=0)
            FP_cumsum = torch.cumsum(fp, dim=0)

            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            
            precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
            precisions = torch.cat((torch.tensor([1]), precisions))
            
            recalls = torch.cat((torch.tensor([0]), recalls))
            
            aps.append(torch.trapz(precisions, recalls))

        return sum(aps) / len(aps)
