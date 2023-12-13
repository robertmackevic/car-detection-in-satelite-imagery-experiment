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
            "total_loss": [],
            "coord_loss": [],
            "object_loss": [],
            "no_object_loss": [],
        }
        self.loss_names = list(self.logs.keys())

    def info(self, message: str) -> None:
        self.logger.info(message)

    def close(self) -> None:
        self.tensorboard_eval.close()
        self.tensorboard_train.close()

    def _clear_logs(self) -> None:
        self.logs = {key: [] for key, _ in self.logs.items()}

    def log_losses(self, losses: Tuple[Tensor, ...]) -> None:
        for loss, loss_name in zip(losses, self.loss_names):
            self.logs[loss_name].append(loss.item())

    def log_epoch(
            self,
            mode: str,
            epoch: Optional[int] = None,
            metrics: Optional[Dict[str, Any]] = None,
            learning_rate: Optional[float] = None
    ) -> None:
        tensorboard = self.tensorboard_train if mode == "train" else self.tensorboard_eval

        message = ""
        for loss, values in self.logs.items():
            mean = sum(values) / len(values)
            message += f"{loss}: {mean:.3f} "
            if epoch is not None:
                tensorboard.add_scalar(tag=f"losses/{loss}", scalar_value=mean, global_step=epoch)

        self.info(message)
        if learning_rate is not None:
            tensorboard.add_scalar(tag="learning_rate", scalar_value=learning_rate, global_step=epoch)

        if metrics is not None:
            message = ""
            for metric, values in metrics.items():
                message += f"{metric}: {values:.3f} "
                if epoch is not None:
                    tensorboard.add_scalar(tag=f"metrics/{metric}", scalar_value=values, global_step=epoch)

            self.info(message)

        self._clear_logs()
