import torch as T
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils import data
from tqdm import tqdm
from model import CNNtoRNN
from config import Hyper, Constants
from coco_data import COCO, COCOData
from collate import Collate
import os
from utils import utils


def validate():
    CUDA_LAUNCH_BLOCKING=1
    ###################### load COCO interface, the input is a json file with annotations ####################
    file_path = os.path.join(Constants.data_folder_ann, Constants.captions_val_file)
    coco_interface = COCO(file_path)
    selected_ann_ids = coco_interface.getAnnIds()
    ####################################################################
    # load ids of images
    # Dataset class takes this list as an input and creates data objects 
    ann_ids = coco_interface.getAnnIds(imgIds=selected_ann_ids)
    ####################################################################
    # selected class ids: extract class id from the annotation
    coco_data_args = {'datalist':ann_ids, 'coco_interface':coco_interface, 'coco_ann_idx':selected_ann_ids, 'stage':'val'}
    coco_data = COCOData(**coco_data_args)
    pad_idx = coco_data.vocab.stoi[Constants.PAD]
    coco_dataloader_args = {'batch_size':Hyper.batch_size, 'shuffle':True, "collate_fn":Collate(pad_idx=pad_idx), "pin_memory":True}
    coco_dataloader = data.DataLoader(coco_data, **coco_dataloader_args)
    step = 0
    # initilze model, loss, etc
    model = CNNtoRNN(coco_data.vocab)
    model = model.to(Constants.device)
    criterion = nn.CrossEntropyLoss(ignore_index=coco_data.vocab.stoi[Constants.PAD])
    #####################################################################
    if Constants.load_model:
        step = load_checkpoint(model, optimizer)

    model.eval()   # Set model to validation mode

    for epoch in range(Hyper.total_epochs):
        print(f"Epoch: {epoch + 1}")
        if Constants.save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for _, (imgs, captions) in tqdm(enumerate(coco_dataloader), total=len(coco_dataloader), leave=False):
            print(captions)
            print(imgs.dtype, "   ", captions.dtype)
            imgs_ = imgs.to(Constants.device)
            captions_ = captions.to(Constants.device)

            outputs = model(imgs_, captions_[:-1])
            outputs1 = outputs.reshape(-1, outputs.shape[2])
            captions1 = captions_.reshape(-1)
            print(outputs1.size(), "    ", captions1)
            # TODO - consider accuracy metrics
            loss = criterion(outputs1, captions1)
            step += 1

if __name__ == "__main__":
    train()