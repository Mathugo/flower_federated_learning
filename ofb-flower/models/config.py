from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """This class contains all the hyperparameters for training the model."""
    lr: float=1e-4
    lr_scheduler: bool=False
    epoch_global: int=2
    epochs: int=2
    batch_size: int=32
    freezing_coeff: float=None
    model: str="mobilenetv3_rw"
    model_stage: str="Staging"
    n_classes: int=3
    data_augmentation: bool=False
    num_workers: int=0
    pin_memory: bool=False
    criterion: str="cross_entropy"