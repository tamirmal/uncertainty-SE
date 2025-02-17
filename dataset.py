import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
import os

# noisy sample: book_02972_chp_0050_reader_02616_19_door_Freesound_validated_235661_0-WP63k5C1qdw-1XtA6tSgFPQ-UEJDB6OMCNY_snr27_fileid_10465.wav
# from the name we need to determine the noise,clean sample
# noise sample: noise_fileid_9.wav
# clean sample: clean_fileid_2509.wav
# 
# Notice the fileid_XXXX at the end ? ...

class SpeechDataset(Dataset):
    def __init__(self, noisy_dir='datasets_generated/noisy', clean_dir='datasets_generated/clean', noise_dir='datasets_generated/noise'):
        self.noisy_files = sorted(os.listdir(noisy_dir))
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.noise_dir = noise_dir

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        fileid = noisy_path.split('_')[-1].split('.')[0]

        clean_path = os.path.join(self.clean_dir, f"clean_fileid_{fileid}.wav")
        noise_path = os.path.join(self.noise_dir, f"noise_fileid_{fileid}.wav")

        noisy, _ = torchaudio.load(noisy_path)
        clean, _ = torchaudio.load(clean_path)
        noise, _ = torchaudio.load(noise_path)
        
        return noisy, clean, noise


# Test the dataset
if __name__ == "__main__":
    dataset = SpeechDataset(
        noisy_dir="/Users/tamirmal/git/DNS_Challenge/datasets_generated/noisy",
        clean_dir="/Users/tamirmal/git/DNS_Challenge/datasets_generated/clean",
        noise_dir="/Users/tamirmal/git/DNS_Challenge/datasets_generated/noise"
        )

    # Calculate the number of files needed for 80 hours and 20 hours
    total_duration_seconds = 80 * 3600 + 20 * 3600  # 80 hours + 20 hours
    file_duration_seconds = 30
    total_files = total_duration_seconds // file_duration_seconds
    train_files = (80 * 3600) // file_duration_seconds
    val_files = total_files - train_files

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_files, val_files])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Print the number of files in each set
    print(f'Training set: {len(train_dataset)} files')
    print(f'Validation set: {len(val_dataset)} files')