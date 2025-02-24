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
hop_length = window_length_samples // 2  # 50% overlap

stft_params_cpu = {
    'n_fft': n_fft,
    'hop_length': hop_length,
    'win_length': window_length_samples,
    'window': torch.hann_window(window_length_samples)
}
stft_params_gpu = {
    'n_fft': n_fft,
    'hop_length': hop_length,
    'win_length': window_length_samples,
    'window': torch.hann_window(window_length_samples)
}

default_hyp = {
    'beta': 0.001
}
###############################################################################

def print_gpu_memory(prefix=''):
    print(f'####:{prefix}')
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")

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

def evaluate_model(model, dataloader, hyperparms = default_hyp):
    beta = hyperparms['beta']
    model.eval()
    running_loss = 0.0
    loss_func = nn.MSELoss()
    with torch.no_grad():
        for noisy, clean, noise in dataloader:
            noisy_stft = torch.stft(noisy.squeeze(1), return_complex=True, **stft_params_cpu)
            clean_stft = torch.stft(clean.squeeze(1), return_complex=True, **stft_params_cpu)
            noisy_mag = torch.abs(noisy_stft)
            x = noisy_mag.permute(0, 2, 1)
            noisy_complex = noisy_stft.permute(0, 2, 1)

            x, noisy_complex = x.to(device), noisy_complex.to(device)
            WF_stft = model(x=x, noisy_complex=noisy_complex)

            WF_stft = WF_stft.permute(0, 2, 1)
            clean, clean_stft = clean.to(device), clean_stft.to(device)

            loss = loss_func(WF_stft, clean_stft)
            # loss is avg over batch, so multiply by batch size to get total loss
            running_loss += loss.item() * noisy.size(0)

    # avg loss over num of samples
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def train_model(model, train_loader, val_loader, num_epochs=25, hyperparms = default_hyp, checkpoint_path='checkpoint.pth'):
    start_epoch = 0
    best_loss = float('inf')
    best_model_path = None
    epochs_no_improve = 0
    beta = hyperparms['beta']

    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    loss_func = nn.MSELoss()

    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)
        print(f'Resuming training from epoch {start_epoch}')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        accumulation_steps = 2  # Accumulate over 2 batches of 32 to get effective batch size 64
        optimizer.zero_grad()  # Zero gradients at the start of each epoch or batch loop
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for batch_idx, (noisy, clean, noise) in enumerate(train_loader):                # 1.torch.STFT expects (B, T) but we have (B, 1, T). so squeeze the channel dimension
                # 2.do this on cpu so we're not holding on gpu ram
                noisy_stft = torch.stft(noisy.squeeze(1), return_complex=True, **stft_params_cpu)
                clean_stft = torch.stft(clean.squeeze(1), return_complex=True, **stft_params_cpu)
                noisy_mag = torch.abs(noisy_stft)
                x = noisy_mag.permute(0, 2, 1)
                noisy_complex = noisy_stft.permute(0, 2, 1)

                # Network input expects: B, 1, T, F => F is last, while torch STFT returns T as last, lets permute
                # and return it back to the original shape after the network for istft
                x, noisy_complex = x.to(device), noisy_complex.to(device)
                WF_stft, AMAP_stft, logvar = model(x=x, noisy_complex=noisy_complex)

                AMAP_stft = AMAP_stft.permute(0, 2, 1)
                WF_stft = WF_stft.permute(0, 2, 1)
                logvar = logvar.permute(0, 2, 1)
                AMAP_istft = torch.istft(AMAP_stft, **stft_params_gpu)
                clean, clean_stft = clean.to(device), clean_stft.to(device)

                loss = loss_func(WF_stft, clean_stft)
                running_loss += loss.item() * noisy.size(0)

                (loss / accumulation_steps).backward()  # Scale loss and accumulate gradients
                # Perform optimizer step every accumulation_steps or at the last batch
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()  # Clear gradients after step
                pbar.update(1)
                # clear cache
                torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Training Loss: {epoch_loss:.4f}')

        # Evaluate on validation set
        val_loss = evaluate_model(model, val_loader)
        print(f'Epoch {epoch}/{num_epochs - 1}, Validation Loss: {val_loss:.4f}')

        # Check if validation loss improved
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model_path = f"/gdrive/MyDrive/Colab Notebooks/speech/mc-dropout/best_model_epoch_{epoch}.pth"
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
#        noisy_dir="/Users/tamirmal/git/DNS_Challenge/datasets_generated/noisy",
#        clean_dir="/Users/tamirmal/git/DNS_Challenge/datasets_generated/clean",
#        noise_dir="/Users/tamirmal/git/DNS_Challenge/datasets_generated/noise"
        noisy_dir="/content/datasets_generated/datasets_generated/noisy",
        clean_dir="/content/datasets_generated/datasets_generated/clean",
        noise_dir="/content/datasets_generated/datasets_generated/noise"
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stft_params_gpu['window']=stft_params_gpu['window'].to(device)
    model = EDNet_uncertainty().to(device)
    best_model_path = train_model(model, train_loader, val_loader, num_epochs=50, checkpoint_path='/gdrive/MyDrive/Colab Notebooks/speech/checkpoint.pth')

    if best_model_path:
        print(f'Best model saved at: {best_model_path}')
