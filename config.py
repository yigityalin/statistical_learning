from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT_DIR / 'dataset'
DATASET_PATH = DATA_DIR / 'diamond.csv'
LOGS_DIR = PROJECT_ROOT_DIR / 'logs'

TARGET = 'Price'


LOGS_DIR.mkdir(exist_ok=True)
