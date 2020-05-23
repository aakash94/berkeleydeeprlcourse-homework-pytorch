import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

import torch

class Agent(nn.Module):

    def __init__(self, input_size = 32, output_size = 32):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size*2)
        self.fc2 = nn.Linear(input_size * 2, input_size*2)
        self.fc3 = nn.Linear(input_size * 2, output_size * 2)
        self.fc4 = nn.Linear(output_size * 2, output_size)


    def forward(self, x):
        #x = torch.FloatTensor(x[0])
        #print(x.size())
        #x = x[0]
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        #x = x.unsqueeze(0)
        return x


    def save_model(self, env_string):
        save_path = "agents/" + env_string + ".chkpt"
        torch.save(self.state_dict(), save_path)


    def load_model(self, env_string):
        save_path = "agents/" + env_string + ".chkpt"
        self.load_state_dict(torch.load(save_path))