from models import Generator, Discriminator
from utils import load_monet_data, load_photo_data
import torch
from pylab import show, imshow, subplot, axis, title
from torchvision.transforms import functional as F


def train(args):
    monet_generator = Generator()
    photo_generator = Generator()

    monet_discriminator = Discriminator()
    photo_discriminator = Discriminator()

    monet_dataset = load_monet_data(batch_size=1)
    photo_dataset = load_photo_data(batch_size=1)

    example_monet = next(iter(monet_dataset))
    example_photo = next(iter(photo_dataset))

    to_monet = monet_generator(example_photo)

    subplot(1,2,1)
    title("Original Photo")
    imshow(F.to_pil_image(example_photo[0] * (256/2) + (256/2)))
    
    subplot(1,2,2)
    title("Monet-esque Photo")
    imshow(F.to_pil_image(to_monet[0] * (256/2) + (256/2)))

    show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='log')
    parser.add_argument('--learningrate', default=0.05)
    parser.add_argument('--epochs', default=5)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()
    train(args)