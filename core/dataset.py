import os
import numpy as np
import torch

from core.data_model import SynData


class Dataset(torch.utils.data.Dataset):

    STD: float
    MEAN: float
    max_delta_time: float

    def __init__(self, npy_files_dir: str, STD: float, MEAN: float, batch_size: int = 8, drop_last=False, max_delta_time=12):
        self.npy_files_dir = npy_files_dir
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.files = os.listdir(self.npy_files_dir)
        self.STD = STD
        self.MEAN = MEAN
        self.max_delta_time =  max_delta_time


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        return SynData.load_from_file(path=os.path.join(self.npy_files_dir, self.files[idx])).to_numpy_array()

    def _normalize(self, element):
        return (element - self.MEAN) / self.STD


    def reprocess(self, data, idxs):
        x = np.array([np.array([
            self._normalize(data[idx][0]), 
            self._normalize(data[idx][1]), 
            self._normalize(data[idx][2]), 
            self._normalize(data[idx][4])])
            for idx in idxs])
        y = np.array([np.array([data[idx][3]]) for idx in idxs]) / self.max_delta_time
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
    return (x.float(), y.float())