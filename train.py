"""
PyTorch training and visualization
Modified by Vu
"""
import sys
import os
sys.setrecursionlimit(1500)


import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torchsummary import summary
from torch.autograd import Variable
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from tqdm import tqdm
import torchnet as tnt
from torch.utils.data import Dataset, DataLoader
import h5py
BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 50

class Weed(Dataset):
    def __init__(self, name):
        hf = h5py.File(name, 'r')
        self.input_images = np.array(hf.get('data')).transpose(0, 3, 1, 2)
        self.target_masks = np.array(hf.get('groundtruth'))
        hf.close()

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        return image, mask

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

def deconvrelu(in_channels, out_channels, kernel, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding, output_padding=1),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    """
    Network description
    """
    def __init__(self, n_filters = 16):
        super(UNet, self).__init__()

        # Contracting Path
        self.conv1 = convrelu(3, n_filters*1, 3, 1)
        self.conv2 = convrelu(n_filters*1, n_filters*2, 3, 1)
        self.conv3 = convrelu(n_filters*2, n_filters*4, 3, 1)
        self.conv4 = convrelu(n_filters*4, n_filters*8, 3, 1)
        self.conv5 = convrelu(n_filters*8, n_filters*16, 3, 1)

        # Expansive Path
        self.conv_up6 = deconvrelu(n_filters*16, n_filters*8, 3, 2, 1)
        self.conv6 = convrelu(2*n_filters*8, n_filters*8, 3, 1)
        self.conv_up7 = deconvrelu(n_filters*8, n_filters*4, 3, 2, 1)
        self.conv7 = convrelu(2*n_filters*4, n_filters*4, 3, 1)
        self.conv_up8 = deconvrelu(n_filters*4, n_filters*2, 3, 2, 1)
        self.conv8 = convrelu(2*n_filters*2, n_filters*2, 3, 1)
        self.conv_up9 = deconvrelu(n_filters*2, n_filters*1, 3, 2, 1)
        self.conv9 = convrelu(2*n_filters*1, n_filters*1, 3, 1)

        self.out = nn.Conv2d(n_filters*1, 1, kernel_size=1, padding=0)

    def forward(self, x, y=None):
        c1 = self.conv1(x)
        p1 = nn.MaxPool2d(2)(c1)

        c2 = self.conv2(p1)
        p2 = nn.MaxPool2d(2)(c2)

        c3 = self.conv3(p2)
        p3 = nn.MaxPool2d(2)(c3)

        c4 = self.conv4(p3)
        p4 = nn.MaxPool2d(2)(c4)
        
        c5 = self.conv5(p4)

        u6 = self.conv_up6(c5)
        cat6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(cat6)

        u7 = self.conv_up7(c6)
        cat7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(cat7)

        u8 = self.conv_up8(c7)
        cat8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(cat8)

        u9 = self.conv_up9(c8)
        cat9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(cat9)

        output = self.out(c9)
        output = F.sigmoid(output)
    
        return output.squeeze()

class LossFunction(nn.Module):
    """
    Loss function
    """
    def __init__(self):
        super(LossFunction, self).__init__()
        self.MSE = nn.MSELoss(size_average=True)

    def forward(self, labels, seg, bce_weight=0.5):
        bce = F.binary_cross_entropy_with_logits(seg, labels)
        # loss = self.MSE(labels, seg)
     
        return bce


if __name__ == "__main__":
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    model = model.to(device)
    model.load_state_dict(torch.load('epochs/epoch_0.pt'))

    summary(model, input_size=(3, 128, 128))
    loss_model = LossFunction()

    ##------------------init------------------------##
    optimizer = Adam(model.parameters())
    
    engine = Engine()#training loop
    meter_loss = tnt.meter.AverageValueMeter()

    dataset_train = Weed('data.h5')
    def get_iterator(mode):
        if mode is True:
            dataset = dataset_train
        elif mode is False:
            dataset = dataset_train
        loader = DataLoader(dataset, batch_size = BATCH_SIZE, num_workers=8, shuffle=mode)

        return loader
    ##------------------log visualization------------------------##
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Images'})
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Segmentations'})
    
    def reset_meters():
        meter_loss.reset()

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        meter_loss.add(state['loss'].item())

    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        print('[Epoch %d] Training Loss: %.4f' % (state['epoch'], meter_loss.value()[0]))

        train_loss_logger.log(state['epoch'], meter_loss.value()[0])

        reset_meters()
        torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % 0)
       
        # visualization.

        test_sample = next(iter(get_iterator(False)))

        ground_truth = (test_sample[0])
        segmentation = model(Variable(ground_truth).type(torch.FloatTensor).cuda())
        seg = segmentation.cpu().data
        seg = seg.reshape((BATCH_SIZE, 1, 128, 128))
        
        ground_truth_logger.log(make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5)))
        reconstruction_logger.log(make_grid(seg, nrow=int(BATCH_SIZE ** 0.5)))

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    ##------------------log visualization------------------------##

    ##------------------main flow------------------------##
    def processor(sample):
        data, labels, training = sample
        data = Variable(data).type(torch.FloatTensor).to(device)
        labels = Variable(labels).type(torch.FloatTensor).to(device)

        if training:
            seg = model(data, labels)
        else:
            seg = model(data)
        
        loss = loss_model(labels, seg)

        return loss, seg


    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
    # def test():
    #     path = os.getcwd() + "/test/result/"
    #     data_test = Weed('test.h5')
    #     weed_test = torch.from_numpy(data_test.input_images).float().to(device)
    #     segmentation = model(Variable(weed_test))
    #     result = segmentation.cpu().data.numpy()
    #     n, h, w = result.shape
    #     for i in range(n):
    #         cv2.imwrite(path + str(i) + ".png", result[i]*255)
    # test()
        

