import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import optim
from time import time
from torchvision.transforms import InterpolationMode

from config import Config
from dataset import SiameseNetworkDataset
from siameseNetwork import SiameseNetwork
from triplet import TripletLoss


def main():
    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    # Set the device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    folder_dataset = dset.ImageFolder(root=Config.training_dir)

    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([
                                                transforms.Grayscale(num_output_channels=3),
                                                transforms.Resize((244, 244)),
                                                transforms.ColorJitter(brightness=(0.5, 1.5),
                                                                       contrast=(0.3, 2.0),
                                                                       hue=.05,
                                                                       saturation=(.0, .15)),

                                                transforms.RandomAffine(0, translate=(0, 0.3),
                                                                        scale=(0.6, 1.8),
                                                                        shear=(0.0, 0.4),
                                                                        interpolation=InterpolationMode.NEAREST,
                                                                        fill=0),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225]),
                                            ]),
                                            should_invert=False)

    print('Loading train dataloader. . .')
    dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=Config.train_batch_size)

    model = nn.DataParallel(SiameseNetwork().to(device))

    print('Model parallelized')

    counter = []
    loss_history = []
    iteration_number = 0
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    prevNum = -1

    for epoch in range(0, Config.train_number_epochs):
        begin = time()
        for batch, data in enumerate(dataloader):
            img_anc, img_pos, img_neg, _ = data
            output1, output2, output3 = model(img_anc.to(device, non_blocking=True), img_pos.to(device, non_blocking=True), img_neg.to(device, non_blocking=True))

            loss = TripletLoss(2)(output1, output2, output3)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if batch % 10 == 0 and prevNum != epoch:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.item())
                prevNum = epoch

        torch.save(model.state_dict(), './res.pth')

        print(time() - begin, 's has passed')

    torch.save(model.state_dict(), './res-300-normalized.pth')


if __name__ == "__main__":
    main()
