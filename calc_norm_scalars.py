import argparse
import numpy as np
from tqdm import tqdm
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, default=None, help="path to dataset dir")
    args = parser.parse_args()

    tmp_list = []
    for item in os.listdir(args.dataset_dir):
        [tmp_list.append(n) for n in np.array(np.load(os.path.join(args.dataset_dir, item)))]
    
    print("std:  " + np.std(np.array(tmp_list)))
    print("mean: " + np.mean(np.array(tmp_list)))