import os
import sys
import torch
import mlflow
import tempfile
import warnings
import subprocess
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from utils.utils import (
    get_run_command,
    get_repository_info,
    generate_confluence_report,
)


def redirect_stdout_to_log(run_dir):
    # Read output file (created by >output/file.txt)
    redirect_file = subprocess.run(
        ["readlink", "-f", f"/proc/{os.getpid()}/fd/1"], capture_output=True, text=True
    ).stdout
    redirect_file = redirect_file[: len(redirect_file) - 1]  # delete \n
    os.remove(redirect_file)

    # Get file to log learning (in dir created by hydra)
    absolute_run_dir = os.path.abspath(run_dir)
    main_log_file = os.path.join(absolute_run_dir, "output.txt")

    # Swap stdout to log file
    f = open(main_log_file, "w")
    sys.stdout = f
    sys.stderr = f

    # Create link to log file (output/file.txt link to log file)
    os.symlink(main_log_file, redirect_file)

    print("Information about files:")
    print(f"File to logging: {main_log_file}")
    print(f"Link file: {redirect_file}")
    return redirect_file


def build_client_participation_histogram(selection_df, num_clients, save_path):
    all_clients = np.concatenate(selection_df["clients"].to_numpy())

    freq = np.bincount(all_clients, minlength=num_clients)

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(num_clients), freq)
    plt.xlabel("Client")
    plt.ylabel("Rounds selected")
    plt.title("Client participation frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class BaseLogger:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.redirect_file = redirect_stdout_to_log(self.run_dir)
        self.run_command = get_run_command()
        print(f"Run command: {self.run_command}\n")

    def end_logging(self):
        self.generate_confluence_report()

    def generate_confluence_report(self):
        if self.checkpoint_path is not None:
            self.report_file = generate_confluence_report(
                self.run_dir, self.checkpoint_path
            )
        else:
            git_info = get_repository_info()
            self.report_file = generate_confluence_report(
                run_dir=self.run_dir, git_info=git_info, run_command=self.run_command
            )
        self.report_file.close()

    def log_run_info(self, cfg):
        pass

    def log_scalar(self, scalar, name, cur_round):
        pass

    def log_pandas(self, pandas, group_name, cur_round):
        pass

    def save_artifact(self, content, artifact_name):
        pass
