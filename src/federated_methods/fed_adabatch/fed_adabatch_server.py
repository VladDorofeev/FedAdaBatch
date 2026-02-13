from ..fedavg.fedavg_server import FedAvgServer

import torch


class FedAdaBatchServer(FedAvgServer):
    def __init__(self, cfg, checkpoint_path):
        super().__init__(cfg)
        self.checkpoint_path = checkpoint_path

    def test_global_model(self):
        if (self.cur_round == 0) and (self.checkpoint_path is not None):
            pretrained_model_state_dict = torch.load(
                self.checkpoint_path,
                weights_only=False,
                map_location=torch.device("cpu"),
            )["model"]
            self.global_model.load_state_dict(pretrained_model_state_dict)
        return super().test_global_model()
