import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
from pathlib import Path
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from unet import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time, sort_path_list
from data import DriveDataset
# from tqdm.notebook import tqdm
from tqdm import tqdm

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    #Progress bar
    pbar = tqdm(enumerate(loader), 'Training', total=len(loader),leave=False)
    
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
        
        # update progressbar
        pbar.set_description(f'Training: (loss {loss.item():.4f})')
        pbar.update()

    epoch_loss = epoch_loss/len(loader)
    
    pbar.close()
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    #Progress bar
    pbar= tqdm(enumerate(loader), 'Validation', total=len(loader),leave=False)
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            
            # update progressbar
            pbar.set_description(f'Validation: (loss {loss.item():.4f})')
            pbar.update()

        epoch_loss = epoch_loss/len(loader)
        
        pbar.close()
    return epoch_loss
            

if __name__ == "__main__":  # Seeding
    seeding(42)

    # Directories
    base_path = Path(__file__).parent.parent
    create_dir(base_path / "checkpoints")

    # Load dataset
    train_x = list((base_path / "new_data/train/images/").glob("*.jpeg"))
    train_y = list((base_path / "new_data/train/masks/").glob("*.jpeg"))
    
    train_x.sort(key=sort_path_list)
    train_y.sort(key=sort_path_list)

    val_x = list((base_path / "new_data/val/images/").glob("*.jpeg"))
    val_y = list((base_path / "new_data/val/masks/").glob("*.jpeg"))
    
    val_x.sort(key=sort_path_list)
    val_y.sort(key=sort_path_list)

    data_str = f"Dataset size:\nTrain: {len(train_x)} - Valid: {len(val_x)}"
    print(data_str)

    # Hyperparameters
    H = 512
    W = 512
    size = (H, W)
    batch_size = 40
    num_epochs = 40
    lr = 1e-4
    checkpoint_path = base_path / "checkpoints/checkpoint.pth"

    # Dataset and Dataloader
    train_dataset = DriveDataset(train_x, train_y)
    val_dataset = DriveDataset(val_x, val_y)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        """ Saving the model """
        if val_loss < best_val_loss:
            data_str = f"Valid loss improved from {best_val_loss:2.4f} to {val_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {val_loss:.3f}\n'
        print(data_str)
