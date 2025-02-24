import torch
import torch.nn as nn

class LpLoss(nn.Module):
    def __init__(self):
        super(LpLoss, self).__init__()

    def forward(self, filtered_output, logvar, clean_speech):
        # Step 1: Compute the variance lambda = exp(logvar)
        lambda_ = torch.exp(logvar)

        # Step 2: Compute the squared error term |S - filtered_output|^2
        error_term = torch.abs(clean_speech - filtered_output) ** 2

        # Step 3: Compute the loss term log(lambda) + (error_term / lambda)
        loss = logvar + (error_term / (lambda_ + 1e-8))  # Add epsilon for numerical stability

        # Step 4: Average the loss across all frequency and time bins, batches
        loss = loss.mean()

        return loss
