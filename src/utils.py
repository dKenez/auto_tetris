import os
from pathlib import Path
import time
import random
import numpy as np
import cv2
import torch

import shutil

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Create a directory. """
def del_new_data():
    new_data_path = Path(__file__).parent.parent / "new_data"
    if os.path.exists(new_data_path):
        print("Deleted new_data directory")
        shutil.rmtree(new_data_path)
    else:
        print("new_data directory doesn't exist")

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == "__main__":
    import sys

    n = len(sys.argv)
    if n == 2:
        if sys.argv[1] == "r":
            del_new_data()