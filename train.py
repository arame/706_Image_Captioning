import torch as T
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils import data
from tqdm import tqdm
from utils import save_checkpoint, save_checkpoint_epoch, load_checkpoint, load_checkpoint_epoch, print_examples
from model import CNNtoRNN
from config import Hyper, Constants
from coco_data import COCO, COCOData
from collate import Collate
import os, sys
from validate import validate
CUDA_LAUNCH_BLOCKING=1

def train():
    file_path_cap = os.path.join(Constants.data_folder_ann, Constants.captions_train_file)
    file_path_inst = os.path.join(Constants.data_folder_ann, Constants.instances_train_file)
    coco_dataloader_train, coco_data_train = get_dataloader(file_path_cap, file_path_inst, "train")
    file_path_cap = os.path.join(Constants.data_folder_ann, Constants.captions_val_file)
    file_path_inst = os.path.join(Constants.data_folder_ann, Constants.instances_val_file)
    coco_dataloader_val, coco_data_val = get_dataloader(file_path_cap, file_path_inst, "val")
    step = 0
    # initilze model, loss, etc
    model = CNNtoRNN(coco_data_train.vocab)
    model = model.to(Constants.device)
    criterion = nn.CrossEntropyLoss(ignore_index=coco_data_train.vocab.stoi[Constants.PAD])
    optimizer = optim.Adam(model.parameters(), lr=Hyper.learning_rate)
    #####################################################################
    if Constants.load_model:
        step = load_checkpoint(model, optimizer)

    model.train()   # Set model to training mode

    for i in range(Hyper.total_epochs):
        epoch = i + 1
        print(f"Epoch: {epoch}")
        if Constants.save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        save_checkpoint_epoch(checkpoint, epoch)
        for _, (imgs, captions) in tqdm(enumerate(coco_dataloader_train), total=len(coco_dataloader_train), leave=False):
            imgs = imgs.to(Constants.device)
            captions = captions.to(Constants.device)
            outputs = model(imgs, captions[:-1])
            vocab_size = outputs.shape[2]
            outputs1 = outputs.reshape(-1, vocab_size)
            captions1 = captions.reshape(-1)
            loss = criterion(outputs1, captions1)
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

        # One epoch's validation
        recent_bleu4 = validate(val_loader=coco_dataloader_val,
                                encoder=model.encoderCNN,
                                decoder=model.decoderRNN,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

def get_dataloader(file_path_cap, file_path_inst, stage):
    ###################### load COCO interface, the input is a json file with annotations ####################
    coco_interface_cap = COCO(file_path_cap)
    coco_interface_inst = COCO(file_path_inst)
    selected_cat_ids = coco_interface_inst.getCatIds(catNms=Hyper.selected_category_names)
    selected_ann_ids = coco_interface_inst.getAnnIds(catIds=selected_cat_ids)
    ####################################################################
    # load ids of images
    # Dataset class takes this list as an input and creates data objects 
    ann_ids = coco_interface_cap.getAnnIds(imgIds=selected_ann_ids)
    print(f"Number of {stage} images = {len(ann_ids)} ")
    if len(ann_ids) == 0:
        sys.exit("Cannot proceed with no images")
    ####################################################################
    # selected class ids: extract class id from the annotation
    coco_data_args = {'datalist':ann_ids, 'coco_interface':coco_interface_cap, 'coco_ann_idx':selected_ann_ids, 'stage':stage}
    coco_data = COCOData(**coco_data_args)
    pad_idx = coco_data.vocab.stoi[Constants.PAD]
    coco_dataloader_args = {'batch_size':Hyper.batch_size, 'shuffle':True, "collate_fn":Collate(pad_idx=pad_idx), "pin_memory":True}
    coco_dataloader = data.DataLoader(coco_data, **coco_dataloader_args)
    return coco_dataloader, coco_data

def train_with_epoch(start_epoch):
    file_path_cap = os.path.join(Constants.data_folder_ann, Constants.captions_train_file)
    file_path_inst = os.path.join(Constants.data_folder_ann, Constants.instances_train_file)
    coco_dataloader_train, coco_data_train = get_dataloader(file_path_cap, file_path_inst, "train")
    file_path_cap = os.path.join(Constants.data_folder_ann, Constants.captions_val_file)
    file_path_inst = os.path.join(Constants.data_folder_ann, Constants.instances_val_file)
    coco_dataloader_val, coco_data_val = get_dataloader(file_path_cap, file_path_inst, "val")
    step = 0
    # initilze model, loss, etc
    model = CNNtoRNN(coco_data_train.vocab)
    model = model.to(Constants.device)
    criterion = nn.CrossEntropyLoss(ignore_index=coco_data_train.vocab.stoi[Constants.PAD])
    optimizer = optim.Adam(model.parameters(), lr=Hyper.learning_rate)
    #####################################################################
    if Constants.load_model:
        step = load_checkpoint(model, optimizer)

    model.train()   # Set model to training mode
    recent_bleu4 = validate(val_loader=coco_dataloader_val,
                                encoder=model.encoderCNN,
                                decoder=model.decoderRNN,
                                criterion=criterion)

    if start_epoch >= Hyper.total_epochs:
        return # Validated the last epoch

    for i in range(start_epoch, Hyper.total_epochs):
        epoch = i + 1
        print(f"Epoch: {epoch}")
        if Constants.save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for _, (imgs, captions) in tqdm(enumerate(coco_dataloader_train), total=len(coco_dataloader_train), leave=False):
            imgs = imgs.to(Constants.device)
            captions = captions.to(Constants.device)
            outputs = model(imgs, captions[:-1])
            vocab_size = outputs.shape[2]
            outputs1 = outputs.reshape(-1, vocab_size)
            captions1 = captions.reshape(-1)
            loss = criterion(outputs1, captions1)
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

        save_checkpoint_epoch(checkpoint, epoch)
        # One epoch's validation
        recent_bleu4 = validate(val_loader=coco_dataloader_val,
                                encoder=model.encoderCNN,
                                decoder=model.decoderRNN,
                                criterion=criterion)
        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

if __name__ == "__main__":
    train()

    ''' Use this method if you want to go straight to the validation and continue from the last saved epoch'''
    #train_with_epoch(1)