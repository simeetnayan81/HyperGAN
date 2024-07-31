
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse

class options:
    def __init__(self, args):
        self.noise_bs=args.noise_batch_size
        self.bs=args.image_batch_size
        self.lr=args.lr
        self.code_size=args.code_size
        self.noise_size=args.noise_size
        self.epochs=args.epochs
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.image_size=[3, 32, 32]

class LeNet5(nn.Module):
    def __init__(self, input_shape):
        super(LeNet5, self).__init__()


        self.pool1=nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool2=nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten=nn.Flatten()


    def forward(self, W, x):

        x=self.pool1(F.relu(F.conv2d(x, W[0], W[1])))
        x=self.pool2(F.relu(F.conv2d(x, W[2], W[3])))
        x=F.conv2d(x, W[4], W[5])
        x=F.relu(x)
        x=self.flatten(x)
        x=F.linear(x, W[6], W[7])
        x=F.relu(x)
        x=F.linear(x, W[8], W[9])
        return x

#Name mixer has been taken from the paper HyperGAN

class Mixer(nn.Module):
    def __init__(self, input_size=256, code_size=128, n_codes=5):
        super(Mixer, self).__init__()
        self.code_size=code_size
        self.n_codes=n_codes
        self.fc1=nn.Linear(in_features=input_size, out_features=128)
        self.fc2=nn.Linear(in_features=128, out_features=256)
        self.fc3=nn.Linear(in_features=256, out_features=self.code_size*self.n_codes)

    def forward(self, x):
        C=self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        C=C.view(-1, self.n_codes, self.code_size)

        c1=C[:, 0, :].view(-1, self.code_size)
        c2=C[:, 1, :].view(-1, self.code_size)
        c3=C[:, 2, :].view(-1, self.code_size)
        c4=C[:, 3, :].view(-1, self.code_size)
        c5=C[:, 4, :].view(-1, self.code_size)

        return [c1, c2, c3, c4, c5]

class Generator(nn.Module):

    def __init__(self, code_size=128, output_size=256):
        super(Generator, self).__init__()
        self.code_size=code_size
        self.output_size=output_size
        self.fc1=nn.Linear(in_features=code_size, out_features=128)
        self.fc2=nn.Linear(in_features=128, out_features=256)
        self.fc3=nn.Linear(in_features=256, out_features=output_size)

        self.output_size = output_size

    def forward(self, code):
        out=self.fc1(code)
        out=F.relu(out)
        out=self.fc2(out)
        out=F.relu(out)
        out=self.fc3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, code_size=256):
        super(Discriminator, self).__init__()
        self.code_size=code_size

        self.fc1=nn.Linear(in_features=self.code_size, out_features=128)
        self.fc2=nn.Linear(in_features=128, out_features=256)
        self.fc3=nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        out=F.sigmoid(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
        return out

#This is model dependent
class LenetGenerator(nn.Module):

    def __init__(self, mixer_input_size, code_size, data_size):
        super(LenetGenerator, self).__init__()
        self.code_size=code_size
        self.n_code=5
        self.mixer_input_size=mixer_input_size
        self.mixer=Mixer(self.mixer_input_size, self.code_size, self.n_code)
        self.g1=Generator(code_size=self.code_size, output_size = 450+6)
        self.g2=Generator(code_size=self.code_size, output_size = 2400+16)
        self.g3=Generator(code_size=self.code_size, output_size = 48000+120)
        self.g4=Generator(code_size=self.code_size, output_size = 9600+80)
        self.g5=Generator(code_size=self.code_size, output_size = 800+10)
        self.lenet=LeNet5(data_size)

    def forward(self, noise):
        return self.forward_generator(self.mixer(noise))

    def forward_lenet(self, W, data):
        return self.lenet(W, data)

    def forward_mixer(self, noise):
        return self.mixer(noise)

    def forward_generator(self, C):

        Z1=self.g1(C[0])
        Z2=self.g2(C[1])
        Z3=self.g3(C[2])
        Z4=self.g4(C[3])
        Z5=self.g5(C[4])

        w1, b1 = Z1[:, :450], Z1[:, -6:]
        w1 = w1.view(-1, 6, 3, 5, 5)
        b1 = b1.view(-1, 6)

        w2, b2 = Z2[:, :2400], Z2[:, -16:]
        w2 = w2.view(-1, 16, 6, 5, 5)
        b2 = b2.view(-1, 16)

        w3, b3 = Z3[:, :48000], Z3[:, -120:]
        w3 = w3.view(-1, 120, 16, 5, 5)
        b3 = b3.view(-1, 120)

        w4, b4 = Z4[:, :9600], Z4[:, -80:]
        w4 = w4.view(-1, 80, 120)
        b4 = b4.view(-1, 80)

        w5, b5 = Z5[:, :800], Z5[:, -10:]
        w5 = w5.view(-1, 10, 80)
        b5 = b5.view(-1, 10)


        return [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5]

class WeightGenerator(nn.Module):
    def __init__(self, noise_size, code_size, data_size):
        super(WeightGenerator, self).__init__()
        self.noise_size=noise_size
        self.code_size=code_size
        self.data_size=data_size
        self.discriminator=Discriminator(self.code_size)
        self.lenetGenerator=LenetGenerator(self.noise_size, self.code_size, self.data_size)

    def forward(self, noise):
        return self.lenetGenerator(noise)

    def train_generator(self, ops, train_dataloader):

        torch.manual_seed(42)
        loss_fn = nn.CrossEntropyLoss() # this is also called "criterion"/"cost function" in some places
        optimizer_gen_mixer = torch.optim.Adam(params=self.lenetGenerator.parameters(), lr=ops.lr)
        optimizer_disc=torch.optim.Adam(params=self.discriminator.parameters(), lr=ops.lr)
        for epoch in range(ops.epochs):
            print(f"-------\n Epoch: {epoch}\n-------")
            self.train()
            train_loss = 0

            for batch, (X, y) in enumerate(train_dataloader):

                # 1. Forward pass
                s=torch.normal(torch.zeros(ops.noise_bs, ops.noise_size)).to(ops.device)
                codes = self.lenetGenerator.forward_mixer(s)
                weights = self.lenetGenerator.forward_generator(codes)

                avg_d_loss=0
                for code in codes:
                    pi = torch.randn(ops.noise_bs, ops.code_size).to(ops.device)
                    d_pi = self.discriminator(pi)
                    d_code = self.discriminator(code)
                    d_pi_loss = -1 * torch.log((1-d_pi).mean())
                    d_code_loss = -1 * torch.log(d_code.mean())
                    d_loss = d_pi_loss + d_code_loss
                    avg_d_loss+=d_loss.item()
                    d_loss.backward(retain_graph=True)
                avg_d_loss/=5
                optimizer_disc.step()

                losses = []
                for (W) in zip(*weights):
                    y_pred = self.lenetGenerator.forward_lenet(W, X.to(ops.device))
                    loss = loss_fn(y_pred, y.to(ops.device))
                    losses.append(loss)

                loss = torch.stack(losses).mean()
                loss.backward()

                optimizer_gen_mixer.step()
                optimizer_disc.zero_grad()
                optimizer_gen_mixer.zero_grad()

                train_loss += loss.item()

                if batch%500==0:
                    print(f"Batch: {batch}, loss: {loss.item()}, discriminator loss: {avg_d_loss}")

            train_loss /= len(train_dataloader)
            print(f"epoch : {epoch}, Average loss {train_loss}")

class EnsembleNetwork(nn.Module):
    def __init__(self, N, G, F, ops):
        super(EnsembleNetwork, self).__init__()
        self.G=G
        self.F=F
        self.N=N
        self.noise_size=ops.noise_size
        self.ops=ops


    def forward(self, x):
        noise=torch.normal(torch.zeros(self.N, self.noise_size)).to(self.ops.device)
        self.G.eval()
        W=self.G(noise)
        y=0
        for (w) in zip(*W):
            y+=self.F(w, x.to(self.ops.device))
        y/=self.N
        return y

def test(ops, model, test_dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = model(images.to(ops.device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(ops.device)).sum().item()
    return correct/total


def argparser():

    parser = argparse.ArgumentParser(description='Weight Generator')
    parser.add_argument('--code_size', default=128, type=int, help='latent code width')
    parser.add_argument('--noise_size', default=64, type=int, help='noise width')
    parser.add_argument('--image_batch_size', default=32, type=int)
    parser.add_argument('--noise_batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--ensemble_size', default='8', type=int)
    parser.add_argument('--epochs', default='10', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args=argparser()
    ops=options(args)
    generator=WeightGenerator(ops.noise_size, ops.code_size, ops.image_size)
    generator.to(ops.device)
    

    train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor(), target_transform=None)


    test_data = datasets.CIFAR10( root="data", train=False, download=True, transform=ToTensor())

    train_dataloader = DataLoader(train_data, batch_size=ops.bs, shuffle=True )

    test_dataloader = DataLoader(test_data, batch_size=ops.bs, shuffle=False )


    print(f"Length of train dataloader: {len(train_dataloader)} batches of {ops.bs}")
    print(f"Length of test dataloader: {len(test_dataloader)} batches of {ops.bs}")

    print("Training Generator")
    generator.train_generator(ops, train_dataloader)
    print("\n\n--------------Training Complete-----------\n")
    lenet=LeNet5(ops.image_size)
    ensemble=EnsembleNetwork(8, generator, lenet, ops)
    print("Size of ensemble : ", ensemble.N)

    acc=test(ops, ensemble, test_dataloader)
    print("Accuracy of ensemble on test dataset: ", acc)