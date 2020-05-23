import os
from torch.utils.data import Dataset
import random

class Loader(Dataset):

    def __init__(self, data_collected):
        self.dataset = data_collected
        # dataset = [ ([],[]), ([],[]), ([],[]), ([].[]) ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        observation, action = self.dataset[index]
        return  observation, action