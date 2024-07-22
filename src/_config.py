import os

BASE_DIR = os.getcwd()  # path to the project directory (the path you run `python` command)

DATA_DIR = f'{BASE_DIR}/data'
LOGS_DIR = f'{BASE_DIR}/logs'
RESULTS_DIR = f"{BASE_DIR}/results"
LOGS_DIR = f'{BASE_DIR}/logs'

# change the name to launch a new W&B project
WANDB_PROJECT = 'ProtoGate'

SEED_VALUE = 42
