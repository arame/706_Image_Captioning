import torch as T
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

#from utils import save_checkpoint, load_checkpoint, print_examples
from model import CNNtoRNN
from config import Hyper, Constants
from coco_data import COCO, COCOData
import os

def train():
    ###################### load COCO interface, the input is a json file with annotations ####################
    file_path = os.path.join(Constants.data_folder_ann, Constants.captions_train_file)
    coco_interface = COCO(file_path)
    selected_ann_ids = coco_interface.getAnnIds()
    ####################################################################
    # load ids of images
    # Dataset class takes this list as an input and creates data objects 
    ann_ids = coco_interface.getAnnIds(imgIds=selected_ann_ids)
    ####################################################################
    # selected class ids: extract class id from the annotation
    coco_data_args = {'datalist':ann_ids, 'coco_interface':coco_interface, 'coco_ann_idx':selected_ann_ids, 'stage':'train'}
    coco_data = COCOData(**coco_data_args)
    #coco_data.test_interface_with_single_image(100)
    coco_dataloader_args = {'batch_size':1, 'shuffle':True}
    coco_dataloader = data.DataLoader(coco_data, **coco_dataloader_args)
    writer = SummaryWriter("runs/coco")
    step = 0
    # initilze model, loss, etc
    model = CNNtoRNN()
    criterion = nn.CrossEntropyLoss()
    #vocab_size = coco_dataloader.dataset
    #####################################################################
    #vocab_size = len(coco_dataloader.dataset.voc())



if __name__ == "__main__":
    train()