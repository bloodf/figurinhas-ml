import torch.nn as nn
from time import time
import torchvision.models as models
from torchvision.models import ResNet101_Weights


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2, progress=True)

    def forward_once(self, x):
        '''
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        #print(output.shape)
        #print(output)
        '''
        b = time()
        output = self.resnet(x)
        print('Time for forward prop: ', time() - b)

        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)

        return output1, output2, output3
