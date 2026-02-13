import time
import copy
import pandas as pd

from utils.data_utils import get_dataset_loader
from ..fedavg.fedavg_client import FedAvgClient


class FedAvgOSClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)

    def _set_train_metrics(self):
        self.train_dataset = self.metrics.to_client_side(self.rank)
        self.init_pos_weight = False

    def _init_loaders(self):
        # No validate data
        self.train_dataset.data.reset_index(inplace=True)
        self.train_loader = get_dataset_loader(
            self.train_dataset, self.cfg, drop_last=False
        )
        self.valid_loader = None

    def get_communication_content(self):
        # In fedavg_client we need to send only result of local learning
        metrics = pd.DataFrame(
            columns=["cifar"],
            index=[
                "Accuracy",
                "Precision",
                "Recall",
                "f1-score",
            ],
        )
        metrics.loc["Accuracy", "cifar"] = 1.0
        metrics.loc["Precision", "cifar"] = 1.0
        metrics.loc["Recall", "cifar"] = 1.0
        metrics.loc["f1-score", "cifar"] = 1.0
        result_dict = {
            "grad": self.grad,
            "rank": self.rank,
            "time": self.result_time,
            "server_metrics": (
                metrics,
                12345,
                100,
            ),
        }
        if self.print_metrics:
            result_dict["client_metrics"] = (self.client_val_loss, self.client_metrics)

        return result_dict

    def train(self):
        start = time.time()

        # Save the server model state to get_grad
        self.server_model_state = copy.deepcopy(self.model).state_dict()

        # No validate data
        # self.server_val_loss, self.server_metrics = self.model_trainer.client_eval_fn(
        #     self
        # )

        # Train client
        self.model_trainer.train_fn(self)

        # Calculate client update
        self.get_grad()

        # Save training time
        self.result_time = time.time() - start
