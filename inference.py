import argparse
import os
import torch
import numpy as np
from dnn.basic_dnn import LinearModel


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, default=None, help="path to model")
    parser.add_argument("--mass", type=float, required=True, default=None, help="")
    parser.add_argument("--cf", type=float, required=True, default=None, help="")
    parser.add_argument("--ambient_temp", type=float, required=True, default=None, help="")
    parser.add_argument("--rectal_temp", type=float, required=True, default=None, help="")
    args = parser.parse_args()

    model = LinearModel()
    model.load_state_dict(torch.load(args.model_path))
    X = torch.nn.functional.normalize(
        torch.from_numpy(
            np.array([np.array([
                args.mass,
                args.cf,
                args.ambient_temp,
                args.rectal_temp
            ])])
        ).float()
    )

    result = model(X, None)
    print(result*12)
