import os
from lightning.pytorch.cli import LightningCLI

from py4cast.datasets import registry as dataset_registry
from py4cast.datasets.base import TorchDataloaderSettings
from py4cast.lightning import (
    ArLightningHyperParam,
    AutoRegressiveLightning,
    PlDataModule,
)
from py4cast.models import registry as model_registry
from py4cast.settings import ROOTDIR

layout = {
    "Check Overfit": {
        "loss": ["Multiline", ["mean_loss_epoch/train", "mean_loss_epoch/validation"]],
    },
}

# Variables for multi-nodes multi-gpu training
nb_nodes = int(os.environ.get("SLURM_NNODES", 1))
if nb_nodes > 1:
    gpus_per_node = len(os.environ.get("SLURM_STEP_GPUS", "1").split(","))
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = global_rank - gpus_per_node * (global_rank // gpus_per_node)
    print(
        f"Global rank: {global_rank}, Local rank: {local_rank}, Gpus per node: {gpus_per_node}"
    )
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["GLOBAL_RANK"] = os.environ.get("SLURM_PROCID", 0)
    os.environ["NODE_RANK"] = os.environ.get("SLURM_NODEID", 0)



def cli_main():
    cli = LightningCLI(AutoRegressiveLightning, PlDataModule)

if __name__ == "__main__":
    cli_main()