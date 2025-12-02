from utils import semisynthetic
import os

DATA_DIR = 'C:/Users/User/OneDrive/Documents/GitHub/sonar_project'

BACKGROUND_DIR = os.path.join(DATA_DIR, "data/klsg_dataset/mine")
OBJECT_DIR = os.path.join(DATA_DIR, "data/masked/plane")
OUTPUT_DIR = os.path.join(DATA_DIR, "synthetic/Procedural/results")

semisynthetic(OUTPUT_DIR, BACKGROUND_DIR, OBJECT_DIR)
