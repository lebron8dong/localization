import time

import numpy as np
import torch
from torch.utils.data import Dataset
import os



class IndoorLocDataSet(Dataset):
    def __init__(self, dataset_path):
        self._csi = []
        self._data_len = 0
        self.domain_label = []
        self.position_label = []
        self.position_label_x = []
        self.position_label_y = []
        self.new_position_label = [[] for _ in range(2)]
        for pa in dataset_path:
            all_data = np.loadtxt(pa, delimiter=',', dtype=np.float32)
            self._data_len += all_data.shape[0]  # RSS指纹数量
            self.csi_len = 234
            csi = torch.from_numpy(all_data[:, :-2]).view(all_data.shape[0], self.csi_len, 4, -1)
            if self._csi == []:
                self._csi = csi
            else:
                self._csi = torch.concat((self._csi, csi), 0)
            # domain_l = all_data[:, -1].reshape(all_data.shape[0])
            for label in all_data[:, -2:]:
                self.domain_label.append(label[1])
                x = label[0] // 18
                y = label[0] % 18
                self.position_label.append(label[0])
                self.position_label_x.append(x)
                self.position_label_y.append(y)
                self.new_position_label.append([x, y])
            # self.domain_label = self.domain_label + domain_l # 域标签
            #
            # self.position_label = self.position_label + all_data[:, -2].reshape(all_data.shape[0]) # 位置标签

            self.co_size = 144  # 位置数量

    def __len__(self):
        return self._data_len

    def __getitem__(self, index):
        csi_item = self._csi[index].tolist()
        # mx = np.matrix(csi_item * self.csi_len, dtype=np.float32).reshape((self.csi_len, self.csi_len, 4)).transpose()
        # for i in range(0, self.csi_len):
        #     mx[:, i, :] = (mx[:, i, :] - csi_item[i, :]) / csi_item[i, :]
        # result = torch.from_numpy(mx.A.reshape(4, self.csi_len, self.csi_len))
        result = self._csi[index].transpose(2, 0)
        return result, self.position_label_x[index], self.position_label_y[index], self.domain_label[index]

# ds = UJIndoorLocDataSet(r"E:\data\dataset", train=False)
# #
# te, pos, domain = ds.__getitem__(10)
# #
# print(te.shape)
