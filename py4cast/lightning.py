import os
import getpass
import shutil
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Tuple, Union

import einops
import matplotlib
import pytorch_lightning as pl
import torch
from lightning.pytorch.utilities import rank_zero_only
from torch import nn
from torchinfo import summary
from transformers import get_cosine_schedule_with_warmup

from py4cast.datasets import get_datasets
from py4cast.datasets.base import (
    DatasetInfo,
    ItemBatch,
    NamedTensor,
    TorchDataloaderSettings,
)
from py4cast.losses import ScaledLoss, WeightedLoss
from py4cast.metrics import MetricACC, MetricPSDK, MetricPSDVar
from py4cast.models import build_model_from_settings, get_model_kls_and_settings
from py4cast.models.base import expand_to_batch
from py4cast.observer import (
    PredictionEpochPlot,
    PredictionTimestepPlot,
    SpatialErrorPlot,
    StateErrorPlot,
)
from py4cast.utils import str_to_dtype

# learning rate scheduling period in steps (update every nth step)
LR_SCHEDULER_PERIOD: int = 10

# PNG plots period in epochs. Plots are made, logged and saved every nth epoch.
PLOT_PERIOD: int = 10

from pytorch_lightning.cli import LightningCLI 
# from lightning.pytorch.cli import LightningCLI

from lightning_fabric.utilities import seed
from py4cast.settings import ROOTDIR
from datetime import datetime
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.profilers import AdvancedProfiler, PyTorchProfiler


from pytorch_lightning import LightningModule, LightningDataModule 
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl 
import torch 
from torch import nn 
from torch.utils.data import DataLoader, Dataset 

# Classe Dataset d'exemple 
class MyDataset(Dataset): 
    def __init__(self, num_samples): 
        # Générer des données d'entrée aléatoires et des cibles 
        self.data = torch.randn(num_samples, 10) # 10 caractéristiques 
        self.labels = torch.randn(num_samples, 1) # 1 cible  
        
    def __len__(self): 
        return len(self.data) 
    def __getitem__(self, idx): 
        return self.data[idx], self.labels[idx] 

class MyDataModule(pl.LightningDataModule): 
    def __init__(self, batch_size = 1, num_samples = 40): 
        super(MyDataModule, self).__init__() 
        self.batch_size = batch_size 
        self.num_samples = num_samples 
        self.dataset = MyDataset(self.num_samples) 
    
    def train_dataloader(self): 
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
 
 # Modèle Lightning 
@dataclass
class MyModel(pl.LightningModule): 
        
    lr: float = 0.01
    batch_size: int= 5
    
    def __post_init__(self):
        super(MyModel, self).__init__() 
        self.layer = nn.Linear(10, 1) # 10 caractéristiques en entrée, 1 en sortie 
    
    def forward(self, x): 
        return self.layer(x) 
    def training_step(self, batch, batch_idx): 
        x, y = batch # Décompose le batch en X et Y 
        y_hat = self.forward(x) 
        loss = nn.functional.mse_loss(y_hat, y) 
        # Calcul de la perte 
        return loss 
    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) 
        return optimizer 

@dataclass
class ExtraArgs:
    campaign_name: str = "camp0"
    run_name: str = "run"
    dev_mode: bool = False
    seed: int = 42
    no_log: bool = False
    profiler: str = "None" # Possibilities are ['simple', 'pytorch', 'None']"
    load_model_ckpt: str = None

class MyCLI(LightningCLI):
    
    def __init__(self, model_class, datamodule_class):
        super().__init__(model_class, datamodule_class)
        # self.model_class = model_class
        # self.datamodule_class = datamodule_class
    
    def add_arguments_to_parser(self, parser):
        parser.add_class_arguments(ExtraArgs, "extra_args")

        parser.link_arguments("data.dataset", "model.hparams.dataset_name")
        parser.link_arguments("data.dataset_conf", "model.hparams.dataset_conf")
        parser.link_arguments("data.dl_settings.batch_size", "model.hparams.batch_size")
        parser.link_arguments("data.num_input_steps", "model.hparams.num_input_steps")
        parser.link_arguments("data.num_pred_steps_train", "model.hparams.num_pred_steps_train")
        parser.link_arguments("data.num_pred_steps_val_test", "model.hparams.num_pred_steps_val_test")
        
        parser.link_arguments("model.hparams.precision", "trainer.precision")
        
        # parser.add_argument("model.hparams.dataset_info")
        # parser.link_arguments("data.train_dataset_info", "model.hparams.dataset_info", apply_on="instantiate")
        pass

    def before_instantiate_classes(self):
        
        extra_args = self.config["fit"]["extra_args"].as_dict()
        data_params = self.config["fit"]["data"].as_dict()
        model_params = self.config["fit"]["model"]["hparams"].as_dict()
        # model_params = self.config["fit"]["model"].as_dict()

        layout = {
            "Check Overfit": {
                "loss": ["Multiline", ["mean_loss_epoch/train", "mean_loss_epoch/validation"]],
            },
        }
        # Variables for multi-nodes multi-gpu training
        self.nb_nodes = int(os.environ.get("SLURM_NNODES", 1))
        if self.nb_nodes > 1:
            gpus_per_node = len(os.environ.get("SLURM_STEP_GPUS", "1").split(","))
            global_rank = int(os.environ.get("SLURM_PROCID", 0))
            local_rank = global_rank - gpus_per_node * (global_rank // gpus_per_node)
            print(
                f"Global rank: {global_rank}, Local rank: {local_rank}, Gpus per node: {gpus_per_node}"
            )
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["GLOBAL_RANK"] = os.environ.get("SLURM_PROCID", 0)
            os.environ["NODE_RANK"] = os.environ.get("SLURM_NODEID", 0)

        self.username = getpass.getuser()
        self.date = datetime.now()

        # Choose seed
        seed.seed_everything(extra_args["seed"])

        # Instantiate dl_settings
        self.dl_settings = TorchDataloaderSettings(
            batch_size=data_params["dl_settings"]["batch_size"],
            num_workers=data_params["dl_settings"]["num_workers"],
            prefetch_factor=data_params["dl_settings"]["prefetch_factor"],
            pin_memory=data_params["dl_settings"]['pin_memory'],
        )

        # Get Log folders
        log_dir = ROOTDIR / "logs"
        folder = Path(extra_args['campaign_name']) / data_params["dataset"] / model_params["model_name"]
        run_name = f"{self.username[:4]}_{extra_args['run_name']}"
        if extra_args["dev_mode"]:
            run_name += "_dev"
        list_subdirs = list((log_dir / folder).glob(f"{run_name}*"))
        list_versions = sorted([int(d.name.split("_")[-1]) for d in list_subdirs])
        version = 0 if list_subdirs == [] else list_versions[-1] + 1
        subfolder = f"{run_name}_{version}"
        self.save_path = log_dir / folder / subfolder

        # Logger & checkpoint callback
        self.callback_list = []
        if extra_args["no_log"]:
            self.logger = None
        else:
            print(
                "--> Model, checkpoints, and tensorboard artifacts "
                + f"will be saved in {self.save_path}."
            )
            self.logger = TensorBoardLogger(
                save_dir=log_dir,
                name=folder,
                version=subfolder,
                default_hp_metric=False,
            )
            self.logger.experiment.add_custom_scalars(layout)
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=self.save_path,
                filename="{epoch:02d}-{val_mean_loss:.2f}",  # Custom filename pattern
                monitor="val_mean_loss",
                mode="min",
                save_top_k=1,  # Save only the best model
                save_last=True,  # Also save the last model
            )
            self.callback_list.append(checkpoint_callback)
            self.callback_list.append(LearningRateMonitor(logging_interval="step"))
            self.callback_list.append(
                EarlyStopping(monitor="val_mean_loss", mode="min", patience=50)
            )

        # Setup profiler
        run_id = self.date.strftime("%b-%d-%Y-%M-%S")
        _profiler = extra_args["profiler"]
        if _profiler == "pytorch":
            self.profiler = PyTorchProfiler(
                dirpath=ROOTDIR / f"logs/{self.config['model']['model']}/{self.config['data']['dataset']}",
                filename=f"torch_profile_{run_id}",
                export_to_chrome=True,
                profile_memory=True,
            )
            print("Initiate pytorchProfiler")
        elif _profiler == "advanced":
            self.profiler = AdvancedProfiler(
                dirpath=ROOTDIR / f"logs/{self.config['model']['model']}/{self.config['data']['dataset']}",
                filename=f"advanced_profile_{run_id}",
                line_count_restriction=50,  # Display top 50 lines
            )
        elif _profiler == "simple":
            self.profiler = _profiler
        else:
            self.profiler = None
            print(f"No profiler set {_profiler}")
    
    def instantiate_classes(self):

        self.config_init = self.parser.instantiate_classes(self.config)

        config = self.config[self.config.subcommand]

        trainer_params = config["trainer"].as_dict()
        data_params = config["data"].as_dict()
        model_params = config["model"]["hparams"].as_dict()
        # model_params = config["model"].as_dict()
        extra_args = config["extra_args"].as_dict()

        #######################################################
        # Datamodule
        #######################################################
        self.datamodule = self.datamodule_class(
            dataset=data_params["dataset"],
            num_input_steps=data_params["num_input_steps"],
            num_pred_steps_train=data_params["num_pred_steps_train"],
            num_pred_steps_val_test=data_params["num_pred_steps_val_test"],
            dl_settings=self.dl_settings,
            dataset_conf=data_params["dataset_conf"],
            config_override=None,
        )

        # # Get essential info to instantiate ArLightningHyperParam
        len_loader = self.datamodule.len_train_dl
        dataset_info = self.datamodule.train_dataset_info

        # Setup GPU usage + get len of loader for LR scheduler
        if torch.cuda.is_available():
            device_name = "cuda"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")  # Allows using Tensor Cores on A100s
            len_loader = len_loader // (torch.cuda.device_count() * self.nb_nodes)
        else:
            device_name = "cpu"
            len_loader = len_loader

        # HP
        hp = ArLightningHyperParam(
            dataset_info=dataset_info,
            dataset_name=model_params['dataset_name'],
            dataset_conf=model_params["dataset_conf"],
            batch_size=model_params["batch_size"],
            model_name=model_params["model_name"],
            model_conf=model_params["model_conf"],
            num_input_steps=model_params['num_input_steps'],
            num_pred_steps_train=model_params['num_pred_steps_train'],
            num_pred_steps_val_test=model_params['num_pred_steps_val_test'],
            num_inter_steps=model_params['num_inter_steps'],
            lr=model_params['lr'],
            loss=model_params['loss'],
            training_strategy=model_params["training_strategy"],
            len_train_loader=len_loader,
            save_path=self.save_path,
            use_lr_scheduler=model_params['use_lr_scheduler'],
            precision=model_params['precision'],
            no_log=model_params["no_log"],
            channels_last=model_params["channels_last"]
        )

        ########################################################
        # Modele
        ########################################################
        # self.model = self.model_class(model_params["lr"], model_params["batch_size"])
        if extra_args["load_model_ckpt"]:
            self.model = self.model_class.load_from_checkpoint(
                extra_args["load_model_ckpt"], hparams=hp
            )
        else:
            self.model = self.model_class(hp)

        # A besoin de config_init pour etre lancé
        # self._add_configure_optimizers_method_to_model(self.subcommand)
        
        ########################################################
        # Trainer
        ########################################################
        # self.trainer = self.instantiate_trainer()
        self.trainer = pl.Trainer(
            num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
            devices="auto",
            max_epochs=trainer_params['max_epochs'],
            deterministic=True,
            strategy="ddp",
            accumulate_grad_batches=10,
            accelerator=device_name,
            logger=self.logger,
            profiler=self.profiler,
            log_every_n_steps=1,
            callbacks=self.callback_list,
            check_val_every_n_epoch=trainer_params["check_val_every_n_epoch"],
            precision=trainer_params["precision"],
            limit_train_batches=trainer_params["limit_train_batches"],
            limit_val_batches=trainer_params["limit_train_batches"],  # No reason to spend hours on validation if we limit the training.
            limit_test_batches=trainer_params["limit_train_batches"],
        )
        pass

@dataclass
class PlDataModule(pl.LightningDataModule):
    """
    DataModule to encapsulate data splits and data loading.
    """

    dataset: str
    num_input_steps: int
    num_pred_steps_train: int
    num_pred_steps_val_test: int
    dl_settings: TorchDataloaderSettings
    dataset_conf: Union[Path, None] = None
    config_override: Union[Dict, None] = None

    def __post_init__(self):
        super().__init__()
        
        # Get dataset in initialisation to have access to this attribute before method trainer.fit
        self.train_ds, self.val_ds, self.test_ds = get_datasets(
            self.dataset,
            self.num_input_steps,
            self.num_pred_steps_train,
            self.num_pred_steps_val_test,
            self.dataset_conf,
            self.config_override,
        )

    @property
    def len_train_dl(self):
        return len(self.train_ds.torch_dataloader(self.dl_settings))

    @property
    def train_dataset_info(self):
        return self.train_ds.dataset_info

    @property
    def infer_ds(self):
        return self.test_ds

    def train_dataloader(self):
        return self.train_ds.torch_dataloader(self.dl_settings)

    def val_dataloader(self):
        return self.val_ds.torch_dataloader(self.dl_settings)

    def test_dataloader(self):
        return self.test_ds.torch_dataloader(self.dl_settings)

    def predict_dataloader(self):
        return self.test_ds.torch_dataloader(self.dl_settings)

@dataclass
class ArLightningHyperParam:
    """
    Settings and hyperparameters for the lightning AR model.
    """

    dataset_info: DatasetInfo = None
    dataset_name: str = "poesy"
    dataset_conf: Path = None
    
    batch_size: int = 7

    model_conf: Union[Path, None] = None
    model_name: str = "halfunet"

    lr: float = 0.1
    loss: str = "mse"

    num_input_steps: int = 2
    num_pred_steps_train: int = 2
    num_inter_steps: int = 1  # Number of intermediary steps (without any data)

    num_pred_steps_val_test: int = 2
    num_samples_to_plot: int = 1

    training_strategy: str = "diff_ar"

    len_train_loader: int = 1
    save_path: Path = None
    use_lr_scheduler: bool = False
    precision: str = "bf16"
    no_log: bool = False
    channels_last: bool = False

    def __post_init__(self):
        """
        Check the configuration

        Raises:
            AttributeError: raise an exception if the set of attribute is not well designed.
        """
        if self.num_inter_steps > 1 and self.num_input_steps > 1:
            raise AttributeError(
                "It is not possible to have multiple input steps when num_inter_steps > 1."
                f"Get num_input_steps :{self.num_input_steps} and num_inter_steps: {self.num_inter_steps}"
            )
        ALLOWED_STRATEGIES = ("diff_ar", "scaled_ar")
        if self.training_strategy not in ALLOWED_STRATEGIES:
            raise AttributeError(
                f"Unknown strategy {self.training_strategy}, allowed strategies are {ALLOWED_STRATEGIES}"
            )

    def summary(self):
        self.dataset_info.summary()
        print(f"Number of input_steps : {self.num_input_steps}")
        print(f"Number of pred_steps (training) : {self.num_pred_steps_train}")
        print(f"Number of pred_steps (test/val) : {self.num_pred_steps_val_test}")
        print(f"Number of intermediary steps :{self.num_inter_steps}")
        print(f"Training strategy :{self.training_strategy}")
        print(
            f"Model step duration : {self.dataset_info.step_duration /self.num_inter_steps}"
        )
        print(f"Model conf {self.model_conf}")
        print("---------------------")
        print(f"Loss {self.loss}")
        print(f"Batch size {self.batch_size}")
        print(f"Learning rate {self.lr}")
        print("---------------------------")


@rank_zero_only
def rank_zero_init(model_kls, model_settings, statics):
    if hasattr(model_kls, "rank_zero_setup"):
        model_kls.rank_zero_setup(model_settings, statics)


@rank_zero_only
def exp_summary(hparams: ArLightningHyperParam, model: nn.Module):
    hparams.summary()
    summary(model)


class AutoRegressiveLightning(pl.LightningModule):
    """
    Auto-regressive lightning module for predicting meteorological fields.
    """

    def __init__(self, hparams: ArLightningHyperParam, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()  # write hparams.yaml in save folder

        # Load static features for grid/data
        # We do not want to change dataset statics inplace
        # Otherwise their is some problem with transform_statics and parameters_saving
        # when relaoding from checkpoint
        statics = deepcopy(hparams.dataset_info.statics)
        # Init object of register_dict
        self.diff_stats = hparams.dataset_info.diff_stats
        self.stats = hparams.dataset_info.stats

        # Keeping track of grid shape
        self.grid_shape = statics.grid_shape

        # For making restoring of optimizer state optional (slight hack)
        self.opt_state = None

        # For example plotting
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

        # class variables to log loss during training, for tensorboad custom scalar
        self.training_step_losses = []
        self.validation_step_losses = []

        # Set model input/output grid features based on dataset tensor shapes
        num_grid_static_features = statics.grid_static_features.dim_size("features")

        # Compute the number of input features for the neural network
        # Should be directly supplied by datasetinfo ?

        num_input_features = (
            hparams.num_input_steps * hparams.dataset_info.weather_dim
            + num_grid_static_features
            + hparams.dataset_info.forcing_dim
        )

        num_output_features = hparams.dataset_info.weather_dim

        model_kls, model_settings = get_model_kls_and_settings(
            hparams.model_name, hparams.model_conf
        )

        # All processes should wait until rank zero
        # has done the initialization (like creating a graph)
        rank_zero_init(model_kls, model_settings, statics)

        self.model, model_settings = build_model_from_settings(
            hparams.model_name,
            num_input_features,
            num_output_features,
            hparams.model_conf,
            statics.grid_shape,
        )
        if hparams.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        exp_summary(hparams, self.model)

        # We transform and register the statics after the model has been set up
        # This change the dimension of all statics
        if len(self.model.input_dims) == 3:
            # Graph model, we flatten the statics spatial dims
            statics.grid_static_features.flatten_("ngrid", 0, 1)
            statics.border_mask = statics.border_mask.flatten(0, 1)
            statics.interior_mask = statics.interior_mask.flatten(0, 1)

        # Register interior and border mask.
        statics.register_buffers(self)

        self.num_spatial_dims = statics.grid_static_features.num_spatial_dims

        self.register_buffer(
            "grid_static_features",
            expand_to_batch(statics.grid_static_features.tensor, hparams.batch_size),
            persistent=False,
        )
        # We need to instantiate the loss after statics had been transformed.
        # Indeed, the statics used should be in the right dimensions.
        # MSE loss, need to do reduction ourselves to get proper weighting
        if hparams.loss == "mse":
            self.loss = WeightedLoss("MSELoss", reduction="none")
        elif hparams.loss == "mae":
            self.loss = WeightedLoss("L1Loss", reduction="none")
        else:
            raise TypeError(f"Unknown loss function: {hparams.loss}")

        self.loss.prepare(self, statics.interior_mask, hparams.dataset_info)

        save_path = self.hparams["hparams"].save_path
        max_pred_step = self.hparams["hparams"].num_pred_steps_val_test - 1
        if self.logging_enabled:
            self.rmse_psd_plot_metric = MetricPSDVar(pred_step=max_pred_step)
            self.psd_plot_metric = MetricPSDK(save_path, pred_step=max_pred_step)
            self.acc_metric = MetricACC(self.hparams["hparams"].dataset_info)

    @property
    def dtype(self):
        """
        Return the appropriate torch dtype for the desired precision in hparams.
        """
        return str_to_dtype[self.hparams["hparams"].precision]

    @rank_zero_only
    def inspect_tensors(self):
        """
        Prints all tensor parameters and buffers
        of the model with name, shape and dtype.
        """
        # trainable parameters
        for name, param in self.named_parameters():
            print(name, param.shape, param.dtype)
        # buffers
        for name, buffer in self.named_buffers():
            print(name, buffer.shape, buffer.dtype)

    @rank_zero_only
    def log_hparams_tb(self):
        if self.logging_enabled and self.logger:
            hparams = self.hparams["hparams"]
            # Log hparams in tensorboard hparams window
            dict_log = asdict(hparams)
            dict_log["username"] = getpass.getuser()
            self.logger.log_hyperparams(dict_log, metrics={"val_mean_loss": 0.0})
            # Save model & dataset conf as files
            if hparams.dataset_conf is not None:
                shutil.copyfile(
                    hparams.dataset_conf, hparams.save_path / "dataset_conf.json"
                )
            if hparams.model_conf is not None:
                shutil.copyfile(
                    hparams.model_conf, hparams.save_path / "model_conf.json"
                )
            # Write commit and state of git repo in log file
            dest_git_log = hparams.save_path / "git_log.txt"
            out_log = (
                subprocess.check_output(["git", "log", "-n", "1"])
                .strip()
                .decode("utf-8")
            )
            out_status = (
                subprocess.check_output(["git", "status"]).strip().decode("utf-8")
            )
            with open(dest_git_log, "w") as f:
                f.write(out_log)
                f.write(out_status)

    def on_fit_start(self):
        self.log_hparams_tb()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        lr = self.hparams["hparams"].lr
        opt = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.95))
        if self.opt_state:
            opt.load_state_dict(self.opt_state)

        if self.hparams["hparams"].use_lr_scheduler:
            len_loader = self.hparams["hparams"].len_train_loader // LR_SCHEDULER_PERIOD
            epochs = self.trainer.max_epochs
            lr_scheduler = get_cosine_schedule_with_warmup(
                opt, 1000 // LR_SCHEDULER_PERIOD, len_loader * epochs
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return opt

    def _next_x(
        self, batch: ItemBatch, prev_states: NamedTensor, step_idx: int
    ) -> torch.Tensor:
        """
        Build the next x input for the model at timestep step_idx using the :
        - previous states
        - forcing
        - static features
        """
        forcing = batch.forcing.select_dim("timestep", step_idx, bare_tensor=False)
        x = torch.cat(
            [
                prev_states.select_dim("timestep", idx)
                for idx in range(batch.num_input_steps)
            ]
            + [self.grid_static_features[: batch.batch_size], forcing.tensor],
            dim=forcing.dim_index("features"),
        )
        return x

    def _step_diffs(
        self, feature_names: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mean and std of the differences between two consecutive states on the desired device.
        """
        step_diff_std = self.diff_stats.to_list("std", feature_names).to(
            device,
            non_blocking=True,
        )
        step_diff_mean = self.diff_stats.to_list("mean", feature_names).to(
            device, non_blocking=True
        )
        return step_diff_std, step_diff_mean

    def _strategy_params(self) -> Tuple[bool, bool, int]:
        """
        Return the parameters for the desired strategy:
        - force_border
        - scale_y
        - num_inter_steps
        """
        force_border: bool = (
            True if self.hparams["hparams"].training_strategy == "scaled_ar" else False
        )
        scale_y: bool = (
            True if self.hparams["hparams"].training_strategy == "scaled_ar" else False
        )
        # raise if mismatch between strategy and num_inter_steps
        if self.hparams["hparams"].training_strategy == "diff_ar":
            if self.hparams["hparams"].num_inter_steps != 1:
                raise ValueError(
                    "Diff AR strategy requires exactly 1 intermediary step."
                )

        return force_border, scale_y, self.hparams["hparams"].num_inter_steps

    def common_step(
        self, batch: ItemBatch, inference: bool = False
    ) -> Tuple[NamedTensor, NamedTensor]:
        """
        Handling autocast subtelty for mixed precision on GPU and CPU (only bf16 for the later).
        """
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                return self._common_step(batch, inference)
        else:
            if "bf16" in self.trainer.precision:
                with torch.cpu.amp.autocast(dtype=self.dtype):
                    return self._common_step(batch, inference)
            else:
                return self._common_step(batch, inference)

    def _common_step(
        self, batch: ItemBatch, inference: bool = False
    ) -> Tuple[NamedTensor, NamedTensor]:
        """
        Two Autoregressive strategies are implemented here for train, val, test and inference:
        - scaled_ar:
            * Boundary forcing with y_true/true_state
            * Scaled Differential update next_state = prev_state + y * std + mean
            * Intermediary steps for which we have no y_true data

        - diff_ar:
            * No Boundary forcing
            * Differential update next_state = prev_state + y
            * No Intermediary steps

        Derived/Inspired from https://github.com/joeloskarsson/neural-lam/

        In inference mode, we assume batch.outputs is None and we disable output based border forcing.
        """
        force_border, scale_y, num_inter_steps = self._strategy_params()
        # Right now we postpone that we have a single input/output/forcing

        self.original_shape = None

        if len(self.model.input_dims) == 3:
            # Stack original shape to reshape later
            self.original_shape = batch.inputs.tensor.shape
            # Graph model, we flatten the batch spatial dims
            batch.inputs.flatten_("ngrid", *batch.inputs.spatial_dim_idx)

            if not inference:
                batch.outputs.flatten_("ngrid", *batch.outputs.spatial_dim_idx)

            batch.forcing.flatten_("ngrid", *batch.forcing.spatial_dim_idx)

        prev_states = batch.inputs
        prediction_list = []

        # Here we do the autoregressive prediction looping
        # for the desired number of ar steps.

        for i in range(batch.num_pred_steps):
            if not inference:
                border_state = batch.outputs.select_dim("timestep", i)

            if scale_y:
                step_diff_std, step_diff_mean = self._step_diffs(
                    self.output_feature_names
                    if inference
                    else batch.outputs.feature_names,
                    prev_states.device,
                )

            # Intermediary steps for which we have no y_true data
            # Should be greater or equal to 1 (otherwise nothing is done).
            for k in range(num_inter_steps):
                x = self._next_x(batch, prev_states, i)
                # Graph (B, N_grid, d_f) or Conv (B, N_lat,N_lon d_f)
                if self.hparams["hparams"].channels_last:
                    x = x.to(memory_format=torch.channels_last)
                y = self.model(x)

                # We update the latest of our prev_states with the network output
                if scale_y:
                    predicted_state = (
                        # select the last timestep
                        prev_states.select_dim("timestep", -1)
                        + y * step_diff_std
                        + step_diff_mean
                    )
                else:
                    predicted_state = prev_states.select_dim("timestep", -1) + y

                # Overwrite border with true state
                # Force it to true state for all intermediary step
                if not inference and force_border:
                    new_state = (
                        self.border_mask * border_state
                        + self.interior_mask * predicted_state
                    )
                else:
                    new_state = predicted_state

                # Only update the prev_states if we are not at the last step
                if i < batch.num_pred_steps - 1 or k < num_inter_steps - 1:
                    # Update input states for next iteration: drop oldest, append new_state
                    timestep_dim_index = batch.inputs.dim_index("timestep")
                    new_prev_states_tensor = torch.cat(
                        [
                            # Drop the oldest timestep (select all but the first)
                            prev_states.index_select_dim(
                                "timestep",
                                range(1, prev_states.dim_size("timestep")),
                            ),
                            # Add the timestep dimension to the new state
                            new_state.unsqueeze(timestep_dim_index),
                        ],
                        dim=timestep_dim_index,
                    )

                    # Make a new NamedTensor with the same dim and
                    # feature names as the original prev_states
                    prev_states = NamedTensor.new_like(
                        new_prev_states_tensor, prev_states
                    )
            # Append prediction to prediction list only "normal steps"
            prediction_list.append(new_state)

        prediction = torch.stack(
            prediction_list, dim=1
        )  # Stacking is done on time step. (B, pred_steps, N_grid, d_f) or (B, pred_steps, N_lat, N_lon, d_f)

        # In inference mode we use a "trained" module which MUST have the output feature names
        # and the output dim names attributes set.
        if inference:
            pred_out = NamedTensor(
                prediction.type(self.output_dtype),
                self.output_dim_names,
                self.output_feature_names,
            )
        else:
            pred_out = NamedTensor.new_like(
                prediction.type_as(batch.outputs.tensor), batch.outputs
            )
        return pred_out, batch.outputs

    def on_train_start(self):
        self.train_plotters = []

    def _shared_epoch_end(self, outputs: List[torch.Tensor], label: str) -> None:
        """Computes and logs the averaged metrics at the end of an epoch.
        Step shared by training and validation epochs.
        """
        if self.logging_enabled:
            avg_loss = torch.stack([x for x in outputs]).mean()
            tb = self.logger.experiment
            tb.add_scalar(f"mean_loss_epoch/{label}", avg_loss, self.current_epoch)

    def training_step(self, batch: ItemBatch, batch_idx: int) -> torch.Tensor:
        """
        Train on single batch
        """

        # we save the feature names at the first batch
        # to check at inference time if the feature names are the same
        # also useful to build NamedTensor outputs with same feature and dim names
        if batch_idx == 0:
            self.input_feature_names = batch.inputs.feature_names
            self.output_feature_names = batch.outputs.feature_names
            self.output_dim_names = batch.outputs.names
            self.output_dtype = batch.outputs.tensor.dtype

        prediction, target = self.common_step(batch)
        # Compute loss: mean over unrolled times and batch
        batch_loss = torch.mean(self.loss(prediction, target))

        self.training_step_losses.append(batch_loss)

        # Notify every plotters
        if self.logging_enabled:
            for plotter in self.train_plotters:
                plotter.update(self, prediction=self.prediction, target=self.target)

        return batch_loss

    @property
    def logging_enabled(self):
        """
        Check if logging is enabled
        """
        return not self.hparams["hparams"].no_log

    def on_save_checkpoint(self, checkpoint):
        """
        We store our feature and dim names in the checkpoint
        """
        checkpoint["input_feature_names"] = self.input_feature_names
        checkpoint["output_feature_names"] = self.output_feature_names
        checkpoint["output_dim_names"] = self.output_dim_names
        checkpoint["output_dtype"] = self.output_dtype

    def on_load_checkpoint(self, checkpoint):
        """
        We load our feature and dim names from the checkpoint
        """
        self.input_feature_names = checkpoint["input_feature_names"]
        self.output_feature_names = checkpoint["output_feature_names"]
        self.output_dim_names = checkpoint["output_dim_names"]
        self.output_dtype = checkpoint["output_dtype"]

    def predict_step(self, batch: ItemBatch, batch_idx: int) -> torch.Tensor:
        """
        Check if the feature names are the same as the one used during training
        and make a prediction.
        """
        if batch_idx == 0:
            if self.input_feature_names != batch.inputs.feature_names:
                raise ValueError(
                    f"Input Feature names mismatch between training and inference. "
                    f"Training: {self.input_feature_names}, Inference: {batch.inputs.feature_names}"
                )
        return self.forward(batch)

    def forward(self, x: ItemBatch) -> NamedTensor:
        """
        Forward pass of the model
        """
        return self.common_step(x, inference=True)[0]

    def on_train_epoch_end(self):
        outputs = self.training_step_losses
        self._shared_epoch_end(outputs, "train")
        self.training_step_losses.clear()  # free memory

    def on_validation_start(self):
        """
        Add some observers when starting validation
        """
        if self.logging_enabled:
            l1_loss = ScaledLoss("L1Loss", reduction="none")
            l1_loss.prepare(
                self, self.interior_mask, self.hparams["hparams"].dataset_info
            )
            metrics = {"mae": l1_loss}
            save_path = self.hparams["hparams"].save_path
            self.valid_plotters = [
                StateErrorPlot(metrics, prefix="Validation"),
                PredictionTimestepPlot(
                    num_samples_to_plot=1,
                    num_features_to_plot=4,
                    prefix="Validation",
                    save_path=save_path,
                ),
                PredictionEpochPlot(
                    num_samples_to_plot=1,
                    num_features_to_plot=4,
                    prefix="Validation",
                    save_path=save_path,
                ),
            ]

    def _shared_val_test_step(self, batch: ItemBatch, batch_idx, label: str):
        with torch.no_grad():
            prediction, target = self.common_step(batch)

        time_step_loss = torch.mean(self.loss(prediction, target), dim=0)
        mean_loss = torch.mean(time_step_loss)

        if self.logging_enabled:
            # Log loss per timestep
            loss_dict = {
                f"timestep_losses/{label}_step_{step}": time_step_loss[step]
                for step in range(time_step_loss.shape[0])
            }
            self.log_dict(loss_dict, on_epoch=True, sync_dist=True)
            self.log(
                f"{label}_mean_loss",
                mean_loss,
                on_epoch=True,
                sync_dist=True,
                prog_bar=(label == "val"),
            )
        return prediction, target, mean_loss

    def validation_step(self, batch: ItemBatch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target, mean_loss = self._shared_val_test_step(
            batch, batch_idx, "val"
        )
        self.validation_step_losses.append(mean_loss)

        self.val_mean_loss = mean_loss

        if self.logging_enabled:
            # Notify every plotters
            if self.current_epoch % PLOT_PERIOD == 0:
                for plotter in self.valid_plotters:
                    plotter.update(self, prediction=prediction, target=target)
                self.psd_plot_metric.update(prediction, target, self.original_shape)
                self.rmse_psd_plot_metric.update(
                    prediction, target, self.original_shape
                )
                self.acc_metric.update(prediction, target)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """

        if self.logging_enabled:
            # Get dict of metrics' results
            dict_metrics = dict()
            dict_metrics.update(self.psd_plot_metric.compute())
            dict_metrics.update(self.rmse_psd_plot_metric.compute())
            dict_metrics.update(self.acc_metric.compute())
            for name, elmnt in dict_metrics.items():
                if isinstance(elmnt, matplotlib.figure.Figure):
                    self.logger.experiment.add_figure(
                        f"{name}", elmnt, self.current_epoch
                    )
                elif isinstance(elmnt, torch.Tensor):
                    self.log_dict(
                        {name: elmnt},
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

        outputs = self.validation_step_losses
        self._shared_epoch_end(outputs, "validation")

        # free memory
        self.validation_step_losses.clear()

        if self.logging_enabled:
            # Notify every plotters
            if self.current_epoch % PLOT_PERIOD == 0:
                for plotter in self.valid_plotters:
                    plotter.on_step_end(self, label="Valid")

    def on_test_start(self):
        """
        Attach observer when starting test
        """
        if self.logging_enabled:
            metrics = {}
            for torch_loss, alias in ("L1Loss", "mae"), ("MSELoss", "rmse"):
                loss = ScaledLoss(torch_loss, reduction="none")
                loss.prepare(
                    self, self.interior_mask, self.hparams["hparams"].dataset_info
                )
                metrics[alias] = loss

            save_path = self.hparams["hparams"].save_path

            self.test_plotters = [
                StateErrorPlot(metrics, save_path=save_path),
                SpatialErrorPlot(),
                PredictionTimestepPlot(
                    num_samples_to_plot=self.hparams["hparams"].num_samples_to_plot,
                    num_features_to_plot=4,
                    prefix="Test",
                    save_path=save_path,
                ),
            ]

    def test_step(self, batch: ItemBatch, batch_idx):
        """
        Run test on single batch
        """
        prediction, target, _ = self._shared_val_test_step(batch, batch_idx, "test")

        if self.logging_enabled:
            # Notify plotters & metrics
            for plotter in self.test_plotters:
                plotter.update(self, prediction=prediction, target=target)

            self.acc_metric.update(prediction, target)
            self.psd_plot_metric.update(prediction, target, self.original_shape)
            self.rmse_psd_plot_metric.update(prediction, target, self.original_shape)

    @cached_property
    def interior_2d(self) -> torch.Tensor:
        """
        Get the interior mask as a 2d mask.
        Usefull when stored as 1D in statics.
        """
        if self.num_spatial_dims == 1:
            return einops.rearrange(
                self.interior_mask, "(x y) h -> x y h", x=self.grid_shape[0]
            )
        return self.interior_mask

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        """
        if self.logging_enabled:
            self.psd_plot_metric.compute()
            self.rmse_psd_plot_metric.compute()
            self.acc_metric.compute()

            # Notify plotters that the test epoch end
            for plotter in self.test_plotters:
                plotter.on_step_end(self, label="Test")

