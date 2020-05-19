# --------------------------------------------------------------------------------------------------------
# 2019/12/26
# src - configuration.py
# md
# --------------------------------------------------------------------------------------------------------

"""
Manages the configuration of the environemet and the recipe for running the machine.
It reads a yaml files into dataclasses.
"""
from dataclasses import dataclass, field
from typing import List

import yaml
from my_tools.python_tools import now_str


@dataclass
class Config:
    """Dataclass with all configuration parameters."""

    device = 'cuda'
    default_config_file: str = './default_config.yml'
    default_recipe_file: str = './default_recipe.yml'
    temp_report_path: str = '../temp_experiments/'
    tb_path = '../tensorboard/'
    datasets_path: str = '/media/md/Datasets/'

    show_batch_images = True
    show_top_losses = True
    tb_projector = False  # TODO: Doesn't work!!!
    log_pr_curve = True
    lr_scheduler = True
    early_stopping = True
    save_last_checkpoint = True
    save_best_checkpoint = True
    save_confusion_matrix = True
    log_stats = True
    creation_time: str = now_str('yyyymmdd_hhmmss')

    @staticmethod
    def save_default_yaml():
        default_rcp = Config().save_yaml(cfg.default_config_file)

    def save_yaml(self, file=None):
        if file is None: file = f'{rcp.base_path}cfg.yml'
        with open(file, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
        return self

    @classmethod
    def load_yaml(cls, file=None):
        """Load the recipe yaml and returns a Recipe dataclass"""
        if file is None: file = f'{rcp.base_path}cfg.yml'
        try:
            with open(file, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            cfg.__dict__ = config
            return cfg
        except FileNotFoundError:
            print("Config file doesn't exist.")


cfg = Config()


@dataclass()
class Recipe:  # Prescription, Ingredient, ModusOperandi
    """
    A dataclass with all the parameters that might vary from one experiment to the other or from one stage of an experiment
    to the other stage
    """

    @dataclass()
    class transforms:
        topilimage: bool = True
        randomrotation: float = 360
        resize: int = 320
        randomverticalflip: float = None
        randomhorizontalflip: float = None
        # colorjitter: dict = field(default_factory=lambda: {'brightness': 0.6, 'saturation': 0.6, 'contrast': 0.6, 'hue': 0.5})
        colorjitter: dict = None
        randomcrop: int = None
        totensor: bool = True
        # normalize: dict = field(default_factory=lambda: {'mean': [0, ], 'std': [1, ]})
        normalize: dict = field(default_factory=lambda: {'mean': [.485, .456, .406], 'std': [.229, .224, .225]})  # imagenet

    experiment: str = ''
    description = ''
    stage: int = 0
    seed: int = 42

    bs: int = 8 * 64
    lr: float = 3e-3
    lr_frac: List[int] = field(default_factory=lambda: [1, 1])  # By how much the lr will be devided
    max_epochs = 25
    shuffle_batch: bool = True
    transforms: dataclass = transforms()

    creation_time: str = now_str('yyyymmdd_hhmmss')

    @property
    def base_path(self):
        return f'{cfg.temp_report_path}{self.experiment}/{self.creation_time}/'

    @property
    def models_path(self):
        return f'{self.base_path}models/'

    @property
    def src_path(self):
        return f'{self.base_path}src/'

    @property
    def tb_log_path(self):
        return f'{cfg.tb_path}{self.experiment}/{self.creation_time}/'

    @property
    def results_path(self):
        return f'{self.base_path}/results/'

    @staticmethod
    def save_default_yaml():
        default_rcp = Recipe().save_yaml(cfg.default_recipe_file)

    def save_yaml(self, file=None):
        if file is None:
            file = f'{self.base_path}rcp_{rcp.stage}.yml'
        with open(file, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
        return self

    @classmethod
    def load_yaml(cls, file=None):
        """Load the recipe yaml and returns a Recipe dataclass"""
        if file is None:
            file = f'{cls.base_path}rcp_{rcp.stage}.yml'
        try:
            with open(file, 'r') as f:
                recipe = yaml.load(f, Loader=yaml.FullLoader)
            rcp.__dict__ = recipe
            return rcp
        except FileNotFoundError:
            print("Recipe file doesn't exist.")
            raise


rcp = Recipe()

if __name__ == '__main__':
    pass
