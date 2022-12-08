import argparse
import os
import numpy as np
from tqdm import tqdm

from core.dataset import Dataset, to_device
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dnn.basic_dnn import LinearModel, LinearModelLoss
import torch.optim as optim


def prepare_dataset(dataset_dir: str, STD, MEAN, batch_size: int = 8, shuffle=False):
    train_dataset = Dataset(npy_files_dir=dataset_dir, batch_size=batch_size, STD=STD, MEAN=MEAN)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=train_dataset.collate_fn,
    )
    return train_dataloader



def train(epochs, dataset_dir, STD, MEAN, batch_size=2048, logdir="logs/", model_dir="models/"):
    if not os.path.isdir(model_dir): os.mkdir(model_dir)
    if not os.path.isdir(logdir): os.mkdir(logdir)

    loss = LinearModelLoss() #.to("cuda")
    model = LinearModel() #.to("cuda")

    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_dataloader = prepare_dataset(dataset_dir=dataset_dir, STD=STD, MEAN=MEAN, batch_size=batch_size, shuffle=True)
    step_counter: int = 0
    best_loss: int = 100 # dummy high score

    for epoch in range(0, epochs):
        for batchs in train_dataloader:
            for batch in batchs:
                optimizer.zero_grad()
                batch = to_device(batch)
                output = model(*batch)
                total_loss = loss(*batch, output)
                total_loss.backward()
                optimizer.step()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                step_counter += 1

                if step_counter % 100 == 0:
                    print(f"Epoch: {epoch} --- Step: {step_counter} --- Loss: {total_loss} --- Max error: {torch.max((output-batch[1]))*12} --- Avg. error: {torch.mean((output-batch[1]))*12}")
        
                if step_counter % 10 == 0:
                    if best_loss > total_loss:
                        best_loss = total_loss
                        #try: 
                        #    os.system("rm -r " + model_dir + "/*")
                        # except: pass

                        torch.save(model.state_dict(), os.path.join(model_dir, str(step_counter)))
                        print(f"model saved, current best loss: {best_loss}")
               


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, default=None, help="path to dataset dir")
    parser.add_argument("--model_dir", type=str, required=False, default="models/", help="path to save models")
    parser.add_argument("--epochs", type=int, required=False, default=100, help="number of epochs to run")
    parser.add_argument("--batch_size", type=int, required=False, default=64, help="number of epochs to run")
    parser.add_argument("--dataset_std", type=float, required=False, default=32.917432, help="")
    parser.add_argument("--dataset_mean", type=float, required=False, default=27.674800, help="")
    args = parser.parse_args()

    train(epochs=args.epochs, STD=args.dataset_std, MEAN=args.dataset_mean, dataset_dir=args.dataset_dir, batch_size=args.batch_size)



