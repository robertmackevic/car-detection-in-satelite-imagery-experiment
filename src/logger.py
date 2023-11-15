import logging
from os import makedirs, listdir
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter


class Logger:
    def __init__(self, tensorboard_dir: Path = Path(".runs")):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

        makedirs(tensorboard_dir, exist_ok=True)
        self.tensorboard_dir = tensorboard_dir / f"v{len(listdir(tensorboard_dir))}"

        self.tensorboard_train = SummaryWriter(log_dir=str(self.tensorboard_dir / "train"))
        self.tensorboard_eval = SummaryWriter(log_dir=str(self.tensorboard_dir / "eval"))

        makedirs(self.tensorboard_train.log_dir, exist_ok=True)
        makedirs(self.tensorboard_eval.log_dir, exist_ok=True)

        self.logs = {
            "loss": [],
            "box_loss": [],
            "object_loss": [],
            "no_object_loss": [],
            "class_loss": [],
        }

    def info(self, message: str) -> None:
        self.logger.info(message)

    def close(self) -> None:
        self.tensorboard_eval.close()
        self.tensorboard_train.close()

    def _clear_logs(self) -> None:
        self.logs = {key: [] for key, _ in self.logs.items()}

    def log_losses(self, losses: Tuple[Tensor, ...]) -> None:
        loss, box_loss, object_loss, no_object_loss, class_loss = losses
        self.logs["loss"].append(loss.item())
        self.logs["box_loss"].append(box_loss.item())
        self.logs["object_loss"].append(object_loss.item())
        self.logs["no_object_loss"].append(no_object_loss.item())
        self.logs["class_loss"].append(class_loss.item())

    def log_epoch(self, mode: str, epoch: Optional[int] = None, metrics: Optional[Dict[str, Any]] = None) -> None:
        tensorboard = self.tensorboard_train if mode == "train" else self.tensorboard_eval

        message = ""
        for loss, values in self.logs.items():
            mean = sum(values) / len(values)
            message += f"{loss}: {mean:.3f} "
            if epoch is not None:
                tensorboard.add_scalar(tag=loss, scalar_value=mean, global_step=epoch)

        self.info(message)

        if metrics is not None:
            message = ""
            for metric, values in metrics.items():
                message += f"{metric}: {values:.3f} "
                if epoch is not None:
                    tensorboard.add_scalar(tag=metric, scalar_value=values, global_step=epoch)

            self.info(message)

        self._clear_logs()
