import torch


class DownBlock(torch.nn.Module):
    def __init__(self, input, output, stride=1, dropout=0, pool=False, apply_norm=True):
        super().__init__()
        self.input = input
        self.output = output
        self.stride = stride
        
        L =[]
        if apply_norm:
            L.append(torch.nn.InstanceNorm2d(num_features=input))
        L.append(torch.nn.LeakyReLU())
        L.append(torch.nn.Dropout2d(dropout))
        if pool:
            L.append(torch.nn.MaxPool2d(kernel_size=3,padding=1,stride=stride))
        L.append(torch.nn.Conv2d(input, output, kernel_size=3, padding=1, stride=stride, bias=False))
        
        self.network = torch.nn.Sequential(*L)
        self.resize = torch.nn.Conv2d(input, output, kernel_size=3, padding=1, stride=stride)

    def forward(self, x):
        residual = x
        x = self.network(x)
        if self.input != self.output or self.stride != 1:
            residual = self.resize(residual)
        return x + residual

class UpBlock(torch.nn.Module):
    def __init__(self, input, output, stride=1, dropout=0, apply_norm=True):
        super().__init__()
        self.input = input
        self.output = output
        self.stride = stride
        
        L =[]
        if apply_norm:
            L.append(torch.nn.InstanceNorm2d(num_features=input))
        L.append(torch.nn.ReLU())
        L.append(torch.nn.Dropout2d(dropout))
        L.append(torch.nn.ConvTranspose2d(input, output, kernel_size=3, padding=1, output_padding=1, stride=stride, bias=False))
        
        self.network = torch.nn.Sequential(*L)

    def forward(self, x):
        x = self.network(x)
        return x

class Generator(torch.nn.Module):
    def __init__(self, layers=[64,128,256,512], channels=3, stride=2):
        super().__init__()

        self.down_stack = [
            DownBlock(channels, layers[0], stride=stride, apply_norm=False), # (bs, 128, 128, 64)
            DownBlock(layers[0], layers[1], stride=stride), # (bs, 64, 64, 128)
            DownBlock(layers[1], layers[2], stride=stride), # (bs, 32, 32, 256)
            DownBlock(layers[2], layers[3], stride=stride), # (bs, 16, 16, 512)
            DownBlock(layers[3], layers[3], stride=stride), # (bs, 8, 8, 512)
            DownBlock(layers[3], layers[3], stride=stride), # (bs, 4, 4, 512)
            DownBlock(layers[3], layers[3], stride=stride), # (bs, 2, 2, 512)
            DownBlock(layers[3], layers[3], stride=stride) # (bs, 1, 1, 512)
        ]

        self.up_stack = [
            UpBlock(layers[3], layers[3], stride=stride, dropout=0.5, apply_norm=False), # (bs, 2, 2, 1024)
            UpBlock(layers[3], layers[3], stride=stride, dropout=0.5), # (bs, 4, 4, 1024)
            UpBlock(layers[3], layers[3], stride=stride, dropout=0.5), # (bs, 8, 8, 1024)
            UpBlock(layers[3], layers[3], stride=stride), # (bs, 16, 16, 1024)
            UpBlock(layers[3], layers[2], stride=stride), # (bs, 32, 32, 512)
            UpBlock(layers[2], layers[1], stride=stride), # (bs, 64, 64, 256)
            UpBlock(layers[1], layers[0], stride=stride) # (bs, 128, 128, 128)
        ]

        L =[]
        L.append(torch.nn.Tanh())
        L.append(torch.nn.ConvTranspose2d(layers[0], channels, kernel_size=3, padding=1, output_padding=1, stride=stride))

        self.last = torch.nn.Sequential(*L) # (bs, 256, 256, 3)

    def forward(self, x):
        skips = []
        for layer in self.down_stack:
            x = layer(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for layer, skip in zip(self.up_stack, skips):
            x = layer(x)
            x = torch.cat([x, skip])
        
        x = self.last(x)
        
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, layers=[64,128,256,512,1], channels=3, stride=2):
        super().__init__()

        L = [
            DownBlock(channels, layers[0], stride=stride, apply_norm=False), # (bs, 128, 128, 64)
            DownBlock(layers[0], layers[1], stride=stride), # (bs, 64, 64, 128)
            DownBlock(layers[1], layers[2], stride=stride), # (bs, 32, 32, 256)
            torch.nn.ZeroPad2d(1), # (bs, 34, 34, 256)
            torch.nn.Conv2d(layers[2], layers[3], kernel_size=3, stride=1, bias=False), # (bs, 31, 31, 512)
            torch.nn.InstanceNorm2d(num_features=layers[3]),
            torch.nn.LeakyReLU(),
            torch.nn.ZeroPad2d(1) # (bs, 33, 33, 512)
        ]

        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Conv2d(layers[3], layers[4], kernel_size=3, stride=1) # (bs, 30, 30, 1)
    
    def forward(self, x):
        out = self.network(x)
        return self.classifier(out)