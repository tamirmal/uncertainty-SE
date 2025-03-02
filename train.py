import os
import sys
import torch
import torchaudio
from torch.utils.data import DataLoader, random_split
from dataset import SpeechDataset
from network import EDNet_uncertainty, EDNet_uncertainty_baseline_wf, EDNet_uncertainty_amap, EDNet_uncertainty_epistemic_dropout, EDNet_uncertainty_wf_logvar
from auraloss.time import SISDRLoss
from LpLoss import LpLoss
from MSELossSpectogram import MSELossSpectrogram
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import math

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
    running_sisdr_loss = 0.0
    running_lp_loss = 0.0
    model_type = model.get_type()

    sisdr_loss_func = SISDRLoss()
    Lp_loss_func = LpLoss()
    mse_loss = MSELossSpectrogram()

    with torch.no_grad():
        for noisy, clean, noise in dataloader:
            noisy_stft = torch.stft(noisy.squeeze(1), return_complex=True, **stft_params_cpu)
            clean_stft = torch.stft(clean.squeeze(1), return_complex=True, **stft_params_cpu)
            noisy_mag = torch.abs(noisy_stft)
            x = noisy_mag.permute(0, 2, 1)
            noisy_complex = noisy_stft.permute(0, 2, 1)

            x, noisy_complex = x.to(device), noisy_complex.to(device)

            if model_type == 'mc-dropout':
                model.enable_dropout(True) # in case it was disabled by .eval()
                wf_stft_samples = []
                for idx in range(model.get_M()):
                    WF_stft, AMAP_stft, logvar = model(x=x, noisy_complex=noisy_complex)
                    wf_stft_samples.append(WF_stft)
                # Stack and average the Wiener filter samples
                wf_stft_samples = torch.stack(wf_stft_samples, dim=0)  # Shape: [mc_iterations, batch, F, T]
                averaged_wf_stft = torch.mean(wf_stft_samples, dim=0)  # Average over MC iterations
                WF_stft = averaged_wf_stft
            else:
                WF_stft, AMAP_stft, logvar = model(x=x, noisy_complex=noisy_complex)

            if AMAP_stft is not None:
                AMAP_stft = AMAP_stft.permute(0, 2, 1)
                AMAP_istft = torch.istft(AMAP_stft, **stft_params_gpu, return_complex=False)
                AMAP_istft = AMAP_istft.unsqueeze(1)

            if logvar is not None:
                logvar = logvar.permute(0, 2, 1)

            if WF_stft is not None:
                WF_stft = WF_stft.permute(0, 2, 1)

            clean, clean_stft = clean.to(device), clean_stft.to(device)

            if model_type in ['aleatoric_amap', 'aleatoric_wf']:
                sisdr_loss = sisdr_loss_func(AMAP_istft, clean)
                Lp_Loss = Lp_loss_func(WF_stft, logvar, clean_stft)
                loss = beta*Lp_Loss + (1.0-beta)*sisdr_loss
                running_loss += loss.item() * noisy.size(0)
                running_sisdr_loss += sisdr_loss.item() * noisy.size(0)
                running_lp_loss += Lp_Loss.item() * noisy.size(0)
            elif model_type in ['baseline_wf', 'mc-dropout']:
                loss = mse_loss(WF_stft, clean_stft)
                running_loss += loss.item() * noisy.size(0)
            elif model_type == 'wf_logvar_Lp':
                Lp_Loss = Lp_loss_func(WF_stft, logvar, clean_stft)
                loss = Lp_Loss
                running_lp_loss += Lp_Loss.item() * noisy.size(0)
                running_loss += loss.item() * noisy.size(0)
            elif model_type == 'amap_sisdr':
                sisdr_loss = sisdr_loss_func(AMAP_istft, clean)
                loss = sisdr_loss
                running_loss += loss.item() * noisy.size(0)
                running_sisdr_loss += sisdr_loss.item() * noisy.size(0)
            elif model_type == 'baseline_wf_sisdr':
                WF_istft = torch.istft(WF_stft, **stft_params_gpu, return_complex=False)
                sisdr_loss = sisdr_loss_func(WF_istft, clean)
                loss = sisdr_loss
                running_sisdr_loss += sisdr_loss.item() * noisy.size(0)
                running_loss += loss.item() * noisy.size(0)

    # avg loss over num of samples
    epoch_loss = running_loss / len(dataloader.dataset)
    sisdr_loss = running_sisdr_loss / len(dataloader.dataset)
    Lp_Loss = running_lp_loss / len(dataloader.dataset)
    return epoch_loss, Lp_Loss, sisdr_loss

def train_model(model, train_loader, val_loader, num_epochs=25, hyperparms = default_hyp, checkpoint_path='checkpoint.pth'):
    start_epoch = 0
    best_loss = float('inf')
    best_model_path = None
    epochs_no_improve = 0
    model_type = model.get_type()

    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    sisdr_loss_func = SISDRLoss()
    Lp_loss_func = LpLoss()
    mse_loss = MSELossSpectrogram()

    if checkpoint_path:
      if os.path.exists(checkpoint_path):
          model, optimizer, start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer)
          print(f'Resuming training from epoch {start_epoch}')
      else:
          assert False, f"Cant find given checkpoint {checkpoint_path}"

    for epoch in range(start_epoch, num_epochs):
        beta = hyperparms['beta']
        model.train()
        running_loss = 0.0
        running_sisdr_loss = 0.0
        running_lp_loss = 0.0
        accumulation_steps = 2  # Accumulate over 2 batches of 32 to get effective batch size 64
        optimizer.zero_grad()  # Zero gradients at the start of each epoch or batch loop
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs - 1}', unit='batch') as pbar:
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

                if AMAP_stft is not None:
                    AMAP_stft = AMAP_stft.permute(0, 2, 1)
                    AMAP_istft = torch.istft(AMAP_stft, **stft_params_gpu, return_complex=False)
                    AMAP_istft = AMAP_istft.unsqueeze(1)
                if logvar is not None:
                    logvar = logvar.permute(0, 2, 1)

                if WF_stft is not None:
                    WF_stft = WF_stft.permute(0, 2, 1)

                clean, clean_stft = clean.to(device), clean_stft.to(device)

                if model_type in ['aleatoric_amap', 'aleatoric_wf']:
                    sisdr_loss = sisdr_loss_func(AMAP_istft, clean)
                    Lp_Loss = Lp_loss_func(WF_stft, logvar, clean_stft)
                    loss = beta*Lp_Loss + (1.0-beta)*sisdr_loss
                    running_loss += loss.item() * noisy.size(0)
                    running_sisdr_loss += sisdr_loss.item() * noisy.size(0)
                    running_lp_loss += Lp_Loss.item() * noisy.size(0)
                elif model_type in ['baseline_wf', 'mc-dropout']:
                    loss = mse_loss(WF_stft, clean_stft)
                    running_loss += loss.item() * noisy.size(0)
                elif model_type == 'wf_logvar_Lp':
                    Lp_Loss = Lp_loss_func(WF_stft, logvar, clean_stft)
                    loss = Lp_Loss
                    running_lp_loss += Lp_Loss.item() * noisy.size(0)
                    running_loss += loss.item() * noisy.size(0)
                elif model_type == 'amap_sisdr':
                    sisdr_loss = sisdr_loss_func(AMAP_istft, clean)
                    loss = sisdr_loss
                    running_loss += loss.item() * noisy.size(0)
                    running_sisdr_loss += sisdr_loss.item() * noisy.size(0)
                elif model_type == 'baseline_wf_sisdr':
                    WF_istft = torch.istft(WF_stft, **stft_params_gpu, return_complex=False)
                    sisdr_loss = sisdr_loss_func(WF_istft, clean)
                    loss = sisdr_loss
                    running_sisdr_loss += sisdr_loss.item() * noisy.size(0)
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
        sisdr_loss = running_sisdr_loss / len(train_loader.dataset)
        Lp_Loss = running_lp_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Training Loss: {epoch_loss:.4f}, SISDR Loss: {sisdr_loss:.4f}, Lp Loss: {Lp_Loss:.4f}')

        # Evaluate on validation set
        val_loss, val_lp_loss, val_sisdr_loss = evaluate_model(model, val_loader, hyperparms={'beta': beta})
        print(f'Epoch {epoch}/{num_epochs - 1}, Validation Loss: {val_loss:.4f}, SISDR Loss: {val_sisdr_loss:.4f}, Lp Loss: {val_lp_loss:.4f}')

        # Check if validation loss improved
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0

            drive_path=os.environ['DRIVE_PATH']
            if model_type == 'mc-dropout':
                best_model_path = f"{drive_path}/{model_type}/{model.get_M()}/best_model_epoch_{epoch}.pth"
            else:
                best_model_path = f"{drive_path}/{model_type}/best_model_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch + 1, val_loss, best_model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve == 10:
            print(f'Early stopping! epoch={epoch}, best_loss={best_loss}')
            break

    return best_model_path

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        model_type = sys.argv[1]
        print(f'Training model: {model_type}')

    drive_path=os.environ['DRIVE_PATH']
    assert drive_path, 'Please set the DRIVE_PATH environment variable'

    if torch.cuda.is_available():
        noisy_dir="/content/noisy"
        clean_dir="/content/clean"
    else:
        noisy_dir="/Users/tamirmal/git/DNS_Challenge/datasets_generated/noisy"
        clean_dir="/Users/tamirmal/git/DNS_Challenge/datasets_generated/clean"

    random.seed(7) # for consistency of dataset split
    dataset = SpeechDataset(
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if torch.cuda.is_available():
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #assert stft_params_cpu == stft_params_gpu # assert equal before moving to device
    stft_params_gpu['window']=stft_params_gpu['window'].to(device)

    if model_type in ['aleatoric_amap', 'aleatoric_wf']:
       # hybrid loss
        model = EDNet_uncertainty(model_type=model_type).to(device)
    elif model_type in ['baseline_wf', 'baseline_wf_sisdr']:
        model = EDNet_uncertainty_baseline_wf(model_type=model_type).to(device)
    elif model_type=='mc-dropout':
        assert len(sys.argv) >= 3, 'MC Dropout requires the number of MC iterations'
        mc_iterations = int(sys.argv[2])
        print(f"mc-dropout={mc_iterations}")
        model = EDNet_uncertainty_epistemic_dropout(M=mc_iterations).to(device)
    elif model_type == 'wf_logvar_Lp':
        # WF with the Lp loss (defined (7) in the paper)
        model = EDNet_uncertainty_wf_logvar(model_type=model_type).to(device)
    elif model_type == 'amap_sisdr':
        # AMAP with the SISDR loss
        model = EDNet_uncertainty_amap(model_type='amap_sisdr').to(device)

    best_model_path = train_model(model, train_loader, val_loader, num_epochs=100, checkpoint_path=None)

    if best_model_path:
        print(f'Best model saved at: {best_model_path}')
