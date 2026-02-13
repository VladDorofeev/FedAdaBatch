import time
import copy
import random

from ..fedavg_one_sample.fedavg_os_client import FedAvgOSClient


class FedAdaBatchClient(FedAvgOSClient):
    def __init__(self, *client_args, **client_kwargs):
        base_client_args = client_args[:2]
        super().__init__(*base_client_args, **client_kwargs)
        self.client_args = client_args

        self.training_type = self.client_args[-2]
        self.local_iters = self.client_args[-1]

    def train_iter_fn(self):
        self.model.train()

        # Split local iterations to:
        # M full epochs
        # S incomplete epochs (S steps for random batches)
        N = len(self.train_loader)
        M, S = divmod(self.local_iters, N)

        # === M fully epochs ===
        for _ in range(M):
            for batch_idx, (_, (inputs, targets)) in enumerate(self.train_loader):
                inp = inputs[0].to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inp)
                loss = self.get_loss_value(outputs, targets)
                loss.backward()
                self.optimizer.step()

        # === S steps for random batches ===
        if S > 0:
            chosen_indices = set(random.sample(range(N), k=S))
            for batch_idx, (_, (inputs, targets)) in enumerate(self.train_loader):
                if batch_idx not in chosen_indices:
                    continue

                inp = inputs[0].to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inp)
                loss = self.get_loss_value(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def train(self):
        assert self.training_type in [
            "rounds",
            "iters",
        ], """Training type should be `rounds` or `iters`."""
        f"""You provide {self.training_type}"""
        if self.training_type == "rounds":
            super().train()
        else:  # self.training_type == "iters"
            start = time.time()
            self.server_model_state = copy.deepcopy(self.model).state_dict()

            # Train client
            self.train_iter_fn()

            self.get_grad()
            self.result_time = time.time() - start
