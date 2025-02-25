import torch
import torch.nn as nn

class MSELossSpectrogram(nn.Module):
    """
    MSE loss for complex spectrograms (STFT) as per Equation (8) in Fang et al. (2023),
    comparing the network's estimated STFT directly with clean STFT ground truth.

    Computes: L = (1/(F*T)) * sum |S_ft - estimated_stft|^2

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            'mean': average over batch (default), 'sum': sum over batch, 'none': no reduction.
    """
    def __init__(self, reduction='mean'):
        super(MSELossSpectrogram, self).__init__()
        self.reduction = reduction

    def forward(self, estimated_stft, clean_stft):
        """
        Compute MSE loss for complex spectrograms.

        Args:
            estimated_stft (torch.Tensor): Network's estimated STFT [batch_size, T, F], complex dtype (torch.complex64)
            clean_stft (torch.Tensor): Clean STFT ground truth [batch_size, T, F], complex dtype (torch.complex64)

        Returns:
            torch.Tensor: Scalar loss (or per-sample losses if reduction='none')
        """
        # Ensure inputs are complex and have correct shape [B, T, F]
        if estimated_stft.dim() != 3 or clean_stft.dim() != 3:
            raise ValueError("Inputs must have shape [batch_size, T, F]")

        if not (estimated_stft.is_complex() and clean_stft.is_complex()):
            raise ValueError("Inputs must be complex tensors (torch.complex64)")

        # Compute squared error: |S_ft - estimated_stft|^2
        squared_error = torch.abs(clean_stft - estimated_stft) ** 2  # [B, T, F], real

        # Average over frequency (F) and time (T) dimensions per sample
        loss = torch.mean(squared_error, dim=[1, 2])  # [B]

        # Apply reduction
        if self.reduction == 'mean':
            loss = torch.mean(loss)  # Average over batch
        elif self.reduction == 'sum':
            loss = torch.sum(loss)  # Sum over batch
        # 'none' returns per-sample losses

        return loss