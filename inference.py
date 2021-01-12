from os import path

from models import Generator, Discriminator
import torch
from torch.autograd import Variable
from torchvision.transforms import functional as F
from pylab import show, imshow, subplot, axis, title
from utils import load_monet_data, load_photo_data


def test(args):
    netG_A2B = Generator().to(args.device)

    checkpoint = torch.load(path.join(path.dirname(path.abspath(__file__)), 'models.th'))
    netG_A2B.load_state_dict(checkpoint['netG_A2B'])

    netG_A2B.eval()

    photo_iterator = iter(load_photo_data(batch_size=args.batch_size))

    Tensor = torch.cuda.FloatTensor if args.device == "cuda" else torch.Tensor
    input_A = Tensor(args.batch_size, 3, 256, 256)

    photo = next(photo_iterator)
    real_A = Variable(input_A.copy_(photo))

    monet = netG_A2B(real_A)[0]
    
    imshow(F.to_pil_image(monet * (256/2) + (256/2)))
    axis('off')
    show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    test(args)