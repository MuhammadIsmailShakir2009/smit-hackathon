"""
ann_model.py
------------
Yeh file hamara Artificial Neural Network (ANN) define karti hai jo
handbook ke mutabiq is project ka PRIMARY PREDICTIVE ENGINE hai (LLM
sirf reasoning/summarization ke liye use hoga, prediction ke liye nahi).

Model ka kaam: 5 tabular features le kar candidate ke "job fit probability"
predict karna (0 = Not Fit, 1 = Fit).
"""

import torch
import torch.nn as nn

from config import FEATURE_NAMES


class ResumeFitANN(nn.Module):
    """
    Simple feed-forward Artificial Neural Network.
    Architecture: Input(5) -> Dense(16) -> ReLU -> Dense(8) -> ReLU -> Dense(1) -> Sigmoid
    """

    def __init__(self, input_dim: int = None):
        super().__init__()
        input_dim = input_dim or len(FEATURE_NAMES)

        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def load_model(model_path, input_dim: int = None) -> ResumeFitANN:
    """Trained model ke weights disk se load karta hai."""
    model = ResumeFitANN(input_dim=input_dim)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model
