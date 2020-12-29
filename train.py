import itertools
from os import path

from models import Generator, Discriminator
from utils import load_monet_data, load_photo_data, LambdaLR, ReplayBuffer
import torch
from pylab import show, imshow, subplot, axis, title
from torchvision.transforms import functional as F
from torch.autograd import Variable
import torch.utils.tensorboard as tb

#https://github.com/aitorzip/PyTorch-CycleGAN
#TODO logger
def train(args):
    netG_A2B = Generator().to(args.device)
    netG_B2A = Generator().to(args.device)

    netD_A = Discriminator().to(args.device)
    netD_B = Discriminator().to(args.device)

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=args.learningrate, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.learningrate, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.learningrate, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.epochs, args.epoch, args.decay_epoch).step)

    Tensor = torch.cuda.FloatTensor if args.device == "cuda" else torch.Tensor
    input_A = Tensor(args.batch_size, 3, 256, 256)
    input_B = Tensor(args.batch_size, 3, 256, 256)
    target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    photo_dataset = load_photo_data(batch_size=args.batch_size)
    monet_dataset = load_monet_data(batch_size=args.batch_size)

    dataloader = zip(photo_dataset, monet_dataset)

    for epoch in range(args.epoch, args.epochs):
        for i, (photo, monet) in enumerate(dataloader):
            real_A = Variable(input_A.copy_(photo))
            real_B = Variable(input_B.copy_(monet))

            optimizer_G.zero_grad()

            # Identity loss
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            
            optimizer_G.step()

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'output/netD_A.pth')
        torch.save(netD_B.state_dict(), 'output/netD_B.pth')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--learningrate', type=float, default=0.05)
    parser.add_argument('--epoch', type=int, default=0) #starting epoch
    parser.add_argument('--decay_epoch', type=int, default=10) #epoch to start linearly decaying the learning rate to 0
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    train(args)