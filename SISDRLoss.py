import torch
import torch.nn as nn

class SISDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio loss implemented as a PyTorch module.

    This loss function:
    1. Performs zero-mean normalization on both target and estimated signals
    2. Calculates the optimal scaling factor for the target
    3. Computes the SI-SDR between the estimated and scaled target signals
    4. Returns the negative mean SI-SDR as the loss value
    """

    def __init__(self, eps=1e-8):
        """
        Initialize the SI-SDR loss module.

        Args:
            eps (float): Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(self, estimated, target):
        """
        Calculate the SI-SDR loss between estimated and target signals.

        Args:
            estimated (Tensor): The estimated signal from the model
            target (Tensor): The target/reference signal

        Returns:
            Tensor: The negative mean SI-SDR value (lower is better)
        """
        # Zero-mean normalization
        target = target - torch.mean(target, dim=-1, keepdim=True)
        estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)

        # Scale-invariant target (find optimal scaling factor)
        alpha = (torch.sum(estimated * target, dim=-1, keepdim=True) /
                (torch.sum(target * target, dim=-1, keepdim=True) + self.eps))
        target_scaled = alpha * target

        # Distortion
        distortion = estimated - target_scaled

        # SI-SDR calculation
        numerator = torch.sum(target_scaled ** 2, dim=-1)
        denominator = torch.sum(distortion ** 2, dim=-1) + self.eps
        si_sdr = 10 * torch.log10(numerator / denominator + self.eps)

        # Return negative mean (to minimize)
        return -si_sdr.mean()