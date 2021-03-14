import os,sys,re
import numpy as np
import tkinter
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import pycocotools
from pycocotools.coco import COCO
import skimage.io as io
from config import Constants, Hyper
from vocabulary import Vocabulary

# dataset interface takes the ids of the COCO classes
class COCOData(data.Dataset):
    def __init__(self, **kwargs):
        self.stage = kwargs['stage']
        self.coco_interface = kwargs['coco_interface']
        # this returns the list of image objects, equal to the number of images of the relevant class(es)
        self.datalist = kwargs['datalist'] 
        # load the list of the image
        self.ann_data = self.coco_interface.loadAnns(self.datalist)
        self.captions = []
        self.image_ids = []
        for i in range(len(self.ann_data)):
            self.captions.append(self.ann_data[i]["caption"])
            self.image_ids.append(self.ann_data[i]["image_id"])
        self.vocab = Vocabulary(Constants.word_threshold)

        if os.path.exists(Constants.vocab_file) & Constants.vocab_from_file:
            self.vocab.get_vocab()
        else:
            self.vocab.build_vocabulary(self.captions)

    # this method normalizes the image and converts it to Pytorch tensor
    # Here we use pytorch transforms functionality, and Compose them together,
    def transform(self, img):
        # these mean values are for RGB!!
        t_ = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            #transforms.Normalize(mean=[0.485, 0.457, 0.407],
                            #                     std=[1,1,1])
                            ])


        img = t_(img)
        # need this for the input in the model
        # returns image tensor (CxHxW)
        return img

    # downloadthe image 
    # return rgb image
    def load_img(self, idx): 
        img_id = self.ann_data[idx]["image_id"]  
        path = self.coco_interface.loadImgs(img_id)   
        coco_url = path[0]["coco_url"]
        im = np.array(io.imread(coco_url))
        im = self.transform(im)
        return im, coco_url

    def load_caption(self, idx):
        caption = self.ann_data[idx]['caption']
        return caption

   
    # number of images
    def __len__(self):
        return len(self.datalist)


    # return image + mask 
    def __getitem__(self, idx):
        img, coco_url = self.load_img(idx)
        caption = self.load_caption(idx)
        numericalized_caption = [self.vocab.stoi[Constants.SOS]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi[Constants.EOS]) 
        return img, coco_url, torch.tensor(numericalized_caption)

    def test_interface_with_single_image(self, image_id):
        ann_ids = self.coco_interface.loadAnns(image_id)
        img, coco_url, caption = self[image_id]
        self.coco_interface.showAnns(ann_ids)
        image = img.squeeze().permute(1,2,0)
        plt.imshow(image)
        plt.savefig("test.png")
        print(f"The image retreived can be found at {coco_url}")
        print("Image saved to test.png. The associated caption is: ")
        print(caption)


""" if __name__ == "__main__":
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
    coco_data.test_interface_with_single_image(59) """

