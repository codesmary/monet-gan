import random

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from os import path, listdir, mkdir
from pylab import show, imshow, subplot, axis
from tfrecord.torch.dataset import MultiTFRecordDataset
from subprocess import call
import numpy as np
import cv2
import torch
from torch.autograd import Variable


def decode_image(features):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=256/2, std=256/2) #scale to [-1,1]
    ])
    image = cv2.imdecode(features["image"], -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image, "RGB")
    image = transform(image)
    return image

def load_data(dataset_path, num_workers=0, batch_size=128):
    idx_files_path = "idx_files"

    if not path.exists(idx_files_path):
        mkdir(idx_files_path)

    tfrec_files = listdir(dataset_path)

    for f in tfrec_files:
        full_path = path.join(dataset_path, f)
        idx_path = path.join(idx_files_path, path.splitext(f)[0] + ".idx")
        if not path.isfile(idx_path):
            call(["python3", "-m", "tfrecord.tools.tfrecord2idx", full_path, idx_path])
    
    tfrecord_pattern = path.join(dataset_path, "{}.tfrec")
    index_pattern = path.join(idx_files_path, "{}.idx")
    splits = {path.splitext(f)[0]: 1/len(tfrec_files) for f in tfrec_files}
    description = {"image_name": "byte", "image": "byte", "target": "byte"}

    dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits, description, transform=decode_image)

    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, drop_last=True)

def load_monet_data(num_workers=0, batch_size=128):
    return load_data(path.join("gan-getting-started", "monet_tfrec"), num_workers, batch_size)

def load_photo_data(num_workers=0, batch_size=128):
    return load_data(path.join("gan-getting-started", "photo_tfrec"), num_workers, batch_size)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

class ReplayBuffer():
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

if __name__ == '__main__':
    dataset = load_monet_data()

    batch = next(iter(dataset))
    for i in range(25):
        im = batch[i]
        subplot(5,5,i+1)
        imshow(F.to_pil_image(im * (256/2) + (256/2)))
        axis('off')
    show()
