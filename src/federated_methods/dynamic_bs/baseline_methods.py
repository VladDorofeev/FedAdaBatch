import warnings
from types import MethodType

from ..fedavg.fedavg import FedAvg
from ..fednova.fednova import FedNova
from ..fedprox.fedprox_client import FedProxClient


def init_federated_fednova(self, cfg):
    FedAvg._init_federated(self, cfg)
    if "SGD" not in str(cfg.optimizer._target_):
        warnings.warn(
            "\nFedNova is designed for use with the SGD optimizer.\n"
            "Behavior with other optimizers is not defined and may be incorrect.\n"
            f"Your optimizer: {cfg.optimizer._target_}",
            UserWarning,
        )


def init_client_cls_fedprox(self):
    FedAvg._init_client_cls(self)
    self.client_cls = FedProxClient
    self.client_kwargs["client_cls"] = self.client_cls
    self.client_args.extend([self.fed_prox_lambda, self.num_fedavg_rounds])


def get_communication_content_fedprox(self, rank):
    content = FedAvg.get_communication_content(self, rank)
    content["current_round"] = self.cur_round
    return content


def configure_aggregation_methods(self, **aggregation_kwargs):
    method = str(getattr(self, "aggregation_method", "fedavg")).lower()
    self.aggregation_method = method

    if method == "fedavg":
        for key, value in aggregation_kwargs.items():
            setattr(self, key, value)
        return self

    if method == "fednova":
        self.calculate_aggregation_weights = MethodType(
            FedNova.calculate_aggregation_weights, self
        )
        self._base_init_federated = MethodType(
            init_federated_fednova, self
        )
        for key, value in aggregation_kwargs.items():
            setattr(self, key, value)
        return self

    if method == "fedprox":
        self.fed_prox_lambda = aggregation_kwargs.pop(
            "fed_prox_lambda", getattr(self, "fed_prox_lambda", 0.0001)
        )
        self.num_fedavg_rounds = aggregation_kwargs.pop(
            "num_fedavg_rounds", getattr(self, "num_fedavg_rounds", 0)
        )
        self._init_client_cls = MethodType(init_client_cls_fedprox, self)
        self.get_communication_content = MethodType(
            get_communication_content_fedprox, self
        )
        for key, value in aggregation_kwargs.items():
            setattr(self, key, value)
        return self

    raise ValueError(
        f"Unsupported aggregation_method='{method}'. "
        "Supported: 'fedavg', 'fedprox', 'fednova'."
    )
