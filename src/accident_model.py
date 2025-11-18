# src/accident_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ACCIDENT_MODEL_PATH

class AccidentMLP(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def load_accident_model(device):
    model = AccidentMLP(in_dim=6, hidden_dim=32).to(device)
    state = torch.load(ACCIDENT_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
