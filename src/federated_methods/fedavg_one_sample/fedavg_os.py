import pandas as pd
from collections import OrderedDict

from ..uniform_fedavg.uniform_fedavg import UniformFedAvg
from .fedavg_os_client import FedAvgOSClient


class FedAvgOS(UniformFedAvg):
    def __init__(self):
        super().__init__()

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedAvgOSClient
        self.client_kwargs["client_cls"] = self.client_cls

    def aggregate(self):
        aggr_weights = super().aggregate()

        # Clear memory after aggregate
        self.server.client_gradients = [
            OrderedDict() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]
        self.server.server_metrics = [
            pd.DataFrame() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]
        return aggr_weights
