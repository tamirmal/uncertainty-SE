import os
import torch
import torchaudio
from torch.utils.data import DataLoader, random_split
from dataset import SpeechDataset
from network import EDNet_uncertainty
from auraloss.time import SISDRLoss
from LpLoss import LpLoss
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

###############################################################################
# "the STFT is computed using a 32 ms Hann window with 50% overlap.""
sample_rate = 16000
window_length_ms = 32  # milliseconds
window_length_samples = int(sample_rate * window_length_ms / 1000)  # = 512 samples
n_fft = window_length_samples  # Usually equal to window length
hop_length = window_length_samples // 4  # 75% overlap -> 128 samples

stft_params = {
    'n_fft': n_fft,
    'hop_length': hop_length,
    'win_length': window_length_samples,
    'window': torch.hann_window(window_length_samples)
}
###############################################################################

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def evaluate_model(model, dataloader, sisdr_loss_func, Lp_loss_func):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for noisy, clean, noise in dataloader:
            noisy, clean, noise = noisy.to(device), clean.to(device), noise.to(device)
            noisy_stft = torch.stft(noisy, **stft_params)
            noisy_mag = torch.abs(noisy_stft)
            WF_stft, AMAP_stft, logvar = model(x=noisy_mag, noisy_complex=noisy_stft)

            AMAP_istft = torch.istft(AMAP_stft, **stft_params)
            clean_istft = torch.istft(clean, **stft_params)
            loss = sisdr_loss_func(AMAP_istft, clean) + Lp_loss_func(WF_stft, logvar, clean_istft)
            running_loss += loss.item() * noisy.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def train_model(model, train_loader, val_loader, num_epochs=25, checkpoint_path='checkpoint.pth'):
    start_epoch = 0
    best_loss = float('inf')
    best_model_path = None
    epochs_no_improve = 0

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    sisdr_loss_func = SISDRLoss()
    Lp_loss_func = LpLoss()

    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)
        print(f'Resuming training from epoch {start_epoch}')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for noisy, clean, noise in train_loader:
                noisy, clean, noise = noisy.to(device), clean.to(device), noise.to(device)
                # torch.STFT expects (B, T) but we have (B, 1, T). so squeeze the channel dimension
                noisy_stft = torch.stft(noisy.squeeze(1), return_complex=True, **stft_params)
                clean_stft = torch.stft(clean.squeeze(1), return_complex=True, **stft_params)
                noisy_mag = torch.abs(noisy_stft)
                optimizer.zero_grad()
                # Network input expects: B, 1, T, F => F is last, while torch STFT returns T as last, lets permute
                # and return it back to the original shape after the network for istft
                x = noisy_mag.permute(0, 2, 1)
                noisy_complex = noisy_stft.permute(0, 2, 1)
                WF_stft, AMAP_stft, logvar = model(x=x, noisy_complex=noisy_complex)
                AMAP_stft = AMAP_stft.permute(0, 2, 1)
                WF_stft = WF_stft.permute(0, 2, 1)
                logvar = logvar.permute(0, 2, 1)
                AMAP_istft = torch.istft(AMAP_stft, **stft_params)
                loss = sisdr_loss_func(AMAP_istft, clean) + Lp_loss_func(WF_stft, logvar, clean_stft)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * noisy.size(0)
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Training Loss: {epoch_loss:.4f}')

        # Evaluate on validation set
        val_loss = evaluate_model(model, val_loader, sisdr_loss_func, Lp_loss_func)
        print(f'Epoch {epoch}/{num_epochs - 1}, Validation Loss: {val_loss:.4f}')

        # Check if validation loss improved
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model_path = f"best_model_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch + 1, val_loss, best_model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve == 10:
            print(f'Early stopping! epoch={epoch}, best_loss={best_loss}')
            break

    return best_model_path

if __name__ == "__main__":
    random.seed(7) # for consistency of dataset split
    dataset = SpeechDataset(
        noisy_dir="/Users/tamirmal/git/DNS_Challenge/datasets_generated/noisy",
        clean_dir="/Users/tamirmal/git/DNS_Challenge/datasets_generated/clean",
        noise_dir="/Users/tamirmal/git/DNS_Challenge/datasets_generated/noise"
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EDNet_uncertainty().to(device)
    best_model_path = train_model(model, train_loader, val_loader, num_epochs=25)

    if best_model_path:
        print(f'Best model saved at: {best_model_path}')
