import os
import numpy as np
import torch

from core.data_model import SynData


class Dataset(torch.utils.data.Dataset):

    def __init__(self, npy_files_dir: str, batch_size: int = 8, drop_last=False):
        self.npy_files_dir = npy_files_dir
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.files = os.listdir(self.npy_files_dir)


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        return SynData.load_from_file(path=os.path.join(self.npy_files_dir, self.files[idx])).to_numpy_array()

    def _normalize(self, element, l):
        return (element - np.mean(l)) / np.std(l)


    def reprocess(self, data, idxs):
        x = np.array([np.array([
            self._normalize(data[idx][0], data[idx]), 
            self._normalize(data[idx][1], data[idx]), 
            self._normalize(data[idx][2], data[idx]), 
            self._normalize(data[idx][4], data[idx])])
            for idx in idxs])
        y = np.array([np.array([data[idx][3]]) for idx in idxs])
        return (x, y)


    def collate_fn(self, data):
        data_size = len(data)
        idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        return output


def to_device(data, device="cpu"):
    (x, y) = data
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    y = y / 12 # max delta time
    return (x.float(), y.float())