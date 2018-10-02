import os

LOGS = './logs/'
DATASETS = './datasets/'
CKPT_ROOT = './checkpoints/'

os.makedirs(LOGS, exist_ok=True)
os.makedirs(CKPT_ROOT, exist_ok=True)
