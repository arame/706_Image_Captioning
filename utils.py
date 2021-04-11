import torch as T
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
from PIL import Image, ImageOps
from config import Constants, Hyper
from model import CNNtoRNN


def print_examples(model, vocab):
    model.eval()
    t_ = None
    if Hyper.is_grayscale:
        t_ = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.485,), (0.456,))
            ])
    else:
        t_ = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 5])
            ])

    test_img1 = get_image("test_examples/fast_train.jpg", t_)
    print("Example 1 CORRECT: fast train travelling across the countryside")
    print("Example 1 OUTPUT: " + " ".join(model.caption_image(test_img1.to(Constants.device))))
    test_img2 = get_image("test_examples/freight_train.jpg", t_)
    print("Example 2 CORRECT: Freight train travelling in front of a mountain")
    print("Example 2 OUTPUT: " + " ".join(model.caption_image(test_img2.to(Constants.device), vocab)))
    test_img3 = get_image("test_examples/inside_a_train.jpg", t_)
    print("Example 3 CORRECT: Passengers inside a train")
    print("Example 3 OUTPUT: " + " ".join(model.caption_image(test_img3.to(Constants.device), vocab)))
    test_img4 = get_image("test_examples/old_train.jpg", t_)
    print("Example 4 CORRECT: Old train travelling under a bridge")
    print("Example 4 OUTPUT: " + " ".join(model.caption_image(test_img4.to(Constants.device), vocab)))
    test_img5 = get_image("test_examples/trains_3.jpg", t_)
    print("Example 5 CORRECT: 3 trains travelling in the same direction")
    print("Example 5 OUTPUT: " + " ".join(model.caption_image(test_img5.to(Constants.device), vocab)))
    test_img6 = get_image("test_examples/train_on_a_bridge.jpg", t_)
    print("Example 6 CORRECT: train on a viaduct")
    print("Example 6 OUTPUT: " + " ".join(model.caption_image(test_img6.to(Constants.device), vocab)))

    """ test_img6 = transform(
        Image.open("test_examples/train_on_a_bridge.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 6 CORRECT: train on a bridge")
    print(
        "Example 6 OUTPUT: "
        + " ".join(model.caption_image(test_img6.to(Constants.device), vocab))
    ) """
    model.train()

def get_image(path, t_):
    temp1 = Image.open(path).convert("RGB")
    temp2 = np.array(temp1)
    temp3 =  t_(temp2)
    temp4 = temp3.unsqueeze(0)
    if Hyper.is_grayscale:
        img = T.cat((temp4, temp4, temp4), 1)
        return img

    img = temp4
    return img

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

if __name__ == "__main__":
    with open(Constants.vocab_file, 'rb') as f:
        vocab = pickle.load(f)
        print('Vocabulary successfully loaded from the vocab.pkl file')
    epoch = 2
    model = CNNtoRNN(vocab)
    model = model.to(Constants.device)
    optimizer = optim.Adam(model.parameters(), lr=Hyper.learning_rate)
    #####################################################################
    _ = load_checkpoint_epoch(model, optimizer, epoch)
    print_examples(model, vocab)