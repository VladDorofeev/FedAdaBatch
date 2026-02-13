import os
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from utils.process_utils import errors_parent_handler


@hydra.main(version_base=None, config_path="../configs", config_name="config")
@errors_parent_handler
def train(cfg: DictConfig):

    os.environ["NO_PROXY"] = (
        "10.100.202.109,10.100.202.109:5000,"
        "10.100.151.14,10.100.151.14:9000,"
        "localhost,127.0.0.1"
    )
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

    # Init federated_method
    trainer = instantiate(cfg.federated_method, _recursive_=False)
    trainer._init_federated(cfg)

    # Add client selection strategy
    client_selector = instantiate(
        trainer.cfg.client_selector, cfg=trainer.cfg, _recursive_=False
    )
    trainer = client_selector(trainer)

    trainer.begin_train()


if __name__ == "__main__":
    train()
