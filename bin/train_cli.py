from lightning.pytorch.cli import LightningCLI

from py4cast.lightning import (
    AutoRegressiveLightning,
    PlDataModule,
)


# install : pip install -U 'jsonargparse[signatures]>=4.27.7'
# Launch : python bin/train_cli.py fit --config config/CLI/test.yaml 

class MyCLI(LightningCLI):
    def __init__(self, model_class, datamodule_class):
        super().__init__(model_class, datamodule_class, save_config_kwargs={"overwrite": True})

    def add_arguments_to_parser(self, parser):
        # parser.add_argument("--save_config_kwargs", type=dict, default={"overwrite": True})
        parser.link_arguments(
            "data.train_dataset_info",
            "model.dataset_info",
            apply_on="instantiate",
        )
    

def cli_main():
    MyCLI(AutoRegressiveLightning, PlDataModule)


if __name__ == "__main__":
    cli_main()