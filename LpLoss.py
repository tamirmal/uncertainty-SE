import torch
import torch.nn as nn

class LpLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(LpLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, filtered_output, logvar, clean_speech):
        # Clamp logvar to prevent extreme values
        logvar_clamped = torch.clamp(logvar, min=-10, max=10)

        # Compute the variance lambda = exp(logvar)
        lambda_ = torch.exp(logvar_clamped)

        # Compute the squared error term
        error_term = torch.abs(clean_speech - filtered_output) ** 2

        # Compute the loss term
        loss = logvar_clamped + (error_term / (lambda_ + self.epsilon))

        # Check for NaN values and handle them
        if torch.isnan(loss).any():
            print("NaN detected in loss computation")
            # You could replace NaN values with a large but finite value
            loss = torch.where(torch.isnan(loss), torch.tensor(100.0, device=loss.device), loss)

        # Average the loss
        loss = loss.mean()

        return loss