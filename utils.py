import torch as T
import torch.optim as optim
import os
from config import Constants, Hyper

def save_checkpoint(checkpoint):
    if os.path.isdir(Constants.backup_model_folder) == False:
        os.mkdir(Constants.backup_model_folder)
    print("=> Saving checkpoint")
    category_path = get_category_filename()
    T.save(checkpoint, category_path)

def save_checkpoint_epoch(checkpoint, epoch):
    epoch_path = get_epoch_filename(epoch)
    if os.path.isdir(Constants.backup_model_folder) == False:
        os.mkdir(Constants.backup_model_folder)
    print(f"=> Saving checkpoint for epoch {epoch}")
    T.save(checkpoint, epoch_path)

def load_checkpoint(model, optimizer):
    print("=> Loading checkpoint")
    category_path = get_category_filename()
    checkpoint = T.load(category_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

def load_checkpoint_epoch(model, optimizer, epoch):
    epoch_path = get_epoch_filename(epoch)
    checkpoint = T.load(epoch_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

def get_category_filename():
    categories = "_".join(Hyper.selected_category_names)
    return f"{Constants.backup_model_path}_{categories}.pth"

def get_epoch_filename(epoch):
    categories = "_".join(Hyper.selected_category_names)
    epoch_path = f"{Constants.backup_model_path}_{categories}_epoch_{epoch}.pth"
    return epoch_path

