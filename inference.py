import argparse
import os
import torch
import numpy as np
from dnn.basic_dnn import LinearModel


def _normalize(element, STD, MEAN):
        return [(e - MEAN) / STD for e in element]


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, default=None, help="path to model")
    parser.add_argument("--mass", type=float, required=True, default=None, help="")
    parser.add_argument("--cf", type=float, required=True, default=None, help="")
    parser.add_argument("--ambient_temp", type=float, required=True, default=None, help="")
    parser.add_argument("--rectal_temp", type=float, required=True, default=None, help="")
    parser.add_argument("--dataset_std", type=float, required=False, default=32.917432, help="")
    parser.add_argument("--dataset_mean", type=float, required=False, default=27.674800, help="")
    parser.add_argument("--result_scalar", type=float, required=False, default=12, help="")
    args = parser.parse_args()

    model = LinearModel()
    model.load_state_dict(torch.load(args.model_path))

    X = torch.from_numpy(
        np.array([
            np.array(
                _normalize([args.mass, args.cf, args.ambient_temp, args.rectal_temp], STD=args.dataset_std, MEAN=args.dataset_mean)
            )
        ])
    ).float()

    result = model(X, None)
    print(result*args.result_scalar)
