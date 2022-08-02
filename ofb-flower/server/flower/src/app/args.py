from dataclasses import dataclass
from os import environ

"""DEFAULTS PARAMETERS"""
SERVER_ADDRESS="0.0.0.0"
ROUNDS=2
# Fraction of available clients used for fit (default: 1)
FRACTION_FIT=1
# Fraction of available clients used for evaluate (default: 0.5)
FRACTION_EVAL=0.5
# Minimum number of clients used for fit/evaluate (default: 2)
MIN_SAMPLE_SIZE=2
#Minimum number of available clients required for sampling (default: 2)
MIN_NUM_CLIENTS=2
#Model Stage to select (default: 'Staging')
MODEL_STAGE='Staging'
#Logserver address (no default)
LOG_HOST=None
#Model to train ("hugonet", "reshugonet", "resnetv2_101x1_bitm", "mobilenetv3_rw", "efficientnet_b0", "vit_base_patch16_224", "vit_small_patch16_224", "vit_small_patch32_224", "vit_tiny_patch16_224")
# The model that is being used for training.
MODEL="mobilenetv3_rw"
#Number of classes
N_CLASSES=3
#Batch size
BATCH_SIZE=32
#Learning rate for client training
LR=1e-4
LR_SCHEDULER=False
#Freezing coefficient to apply to layers when using transfer learning with big models, 0<=freezing_coeff<=1 (default None)
FREEZING_COEFF=None
#Criterion for training and testing
CRITERION="cross_entropy"
#Number of workers for dataset reading
NUM_WORKERS=0
#LOCAL EPOCHS TO PERFORM ON EACH CLIENT
LOCAL_EPOCHS=2
MLFLOW_SERVER_PORT=4040
MLFLOW_SERVER_IP="localhost"
#Data augmentation to apply on client dataset
DATA_AUGMENTATION=False
#If you load your samples in the Dataset on CPU and would like to push it during training to the GPU, you can speed up the host to device transfer by enabling pin_memory.
PIN_MEMORY=False
LOAD_MLFLOW_MODEL=True

# TODO load training config from json
@dataclass
class Args:
    """It's a class that holds the arguments for the program"""
    
    server_address:     str=str(environ.get('SERVER_ADDRESS', SERVER_ADDRESS))
    rounds:             int=int(environ.get('ROUNDS', ROUNDS))
    fraction_fit:       int=int(environ.get('FRACTION_FIT', FRACTION_FIT))
    fraction_eval:      float=float(environ.get('FRACTION_EVAL', FRACTION_EVAL))
    min_sample_size:    int=int(environ.get('MIN_SAMPLE_SIZE', MIN_SAMPLE_SIZE))
    min_num_clients:    int=int(environ.get('MIN_NUM_CLIENTS', MIN_NUM_CLIENTS))
    model_stage:        str=str(environ.get('MODEL_STAGE', MODEL_STAGE))
    log_host:           str=str(environ.get('LOG_HOST', LOG_HOST))
    model:              str=str(environ.get('MODEL', MODEL))
    n_classes:          int=int(environ.get('N_CLASSES', N_CLASSES))
    batch_size:         int=int(environ.get('BATCH_SIZE', BATCH_SIZE))
    lr:                 float=float(environ.get('LR', LR))
    lr_scheduler:       bool=bool(environ.get('LR_SCHEDULER', LR_SCHEDULER))
    criterion:          str=str(environ.get('CRITERION', CRITERION))
    freezing_coeff:     float=environ.get('FREEZING_COEFF', FREEZING_COEFF)
    num_workers:        int=int(environ.get('NUM_WORKERS', NUM_WORKERS))
    local_epochs:       int=int(environ.get('LOCAL_EPOCHS', LOCAL_EPOCHS))
    mlflow_server_port: int=int(environ.get('MLFLOW_PORT', MLFLOW_SERVER_PORT))
    mlflow_server_ip:   str=str(environ.get('MLFLOW_HOST', MLFLOW_SERVER_IP))
    data_augmentation:  bool=bool(environ.get('DATA_AUGMENTATION', DATA_AUGMENTATION))
    pin_memory:         bool=bool(environ.get('PIN_MEMORY', PIN_MEMORY))
    load_mlflow_model:  bool=bool(environ.get('LOAD_MLFLOW_MODEL', LOAD_MLFLOW_MODEL))
