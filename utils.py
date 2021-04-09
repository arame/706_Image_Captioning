import torch as T
import torchvision.transforms as transforms
import os
from PIL import Image
from config import Constants


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("test_examples/dog.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    test_img2 = transform(
        Image.open("test_examples/child.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )
    test_img3 = transform(Image.open("test_examples/bus.png").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: Bus driving by parked cars")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )
    test_img4 = transform(
        Image.open("test_examples/boat.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: A small boat in the ocean")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    )
    test_img5 = transform(
        Image.open("test_examples/horse.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    )
    model.train()


def save_checkpoint(checkpoint):
    if os.path.isdir(Constants.backup_model_folder) == False:
        os.mkdir(Constants.backup_model_folder)
    print("=> Saving checkpoint")
    T.save(checkpoint, Constants.backup_model_path)

def save_checkpoint_epoch(checkpoint, epoch):
    suffix = f"_epoch_{epoch}"
    epoch_path = Constants.backup_model_path + suffix
    if os.path.isdir(Constants.backup_model_folder) == False:
        os.mkdir(Constants.backup_model_folder)
    print(f"=> Saving checkpoint for epoch {epoch}")
    T.save(checkpoint, epoch_path)

def load_checkpoint(model, optimizer):
    print("=> Loading checkpoint")
    checkpoint = T.load(Constants.backup_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

def load_checkpoint_epoch(model, optimizer, epoch):
    suffix = f"_epoch_{epoch}"
    epoch_path = Constants.backup_model_path + suffix
    checkpoint = T.load(epoch_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
