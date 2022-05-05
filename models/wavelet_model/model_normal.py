from models.wavelet_model.conv_block import BasicConv,\
    DAB
import torch.nn as nn
from models.wavelet_model import dct
class NormalModel(nn.Module):
    def __init__(self,in_channel):
        super(NormalModel, self).__init__()
        self.in_channel = in_channel
        # self.dct = dct.dct
        # self.pool = WaveletPool()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.out1 = 8
        self.conv1_1 = BasicConv(in_planes=in_channel,out_planes=self.out1)
        self.conv1_2 = BasicConv(in_planes=self.out1,out_planes=self.out1)

        self.out2 = 16
        self.conv2_1 = BasicConv(in_planes=self.out1, out_planes=self.out2)
        self.conv2_2 = BasicConv(in_planes=self.out2, out_planes=self.out2)
        # self.dab1 = DAB(self.out1,kernel_size=3,reduction=4)

        self.fc_2_1 = nn.Linear(self.out2, self.out2 * 2)
        self.fc_2_2 = nn.Linear(self.out2 * 2, self.out2)
        self.fc_2_3 = nn.Linear(self.out2, 16)

        self.out3 = 32
        self.conv3_1 = BasicConv(in_planes=self.out2,out_planes=self.out3)
        self.conv3_2 = BasicConv(in_planes=self.out3,out_planes=self.out3)
        # self.dab2 = DAB(self.out2,kernel_size=3,reduction=4)
        self.out4 = 64
        self.conv4_1 = BasicConv(in_planes=self.out3,out_planes=self.out4)
        self.conv4_2 = BasicConv(in_planes=self.out4, out_planes=self.out4)
        # self.dab3 = DAB(self.out3,kernel_size=3,reduction=8)
        self.fc_4_1 = nn.Linear(self.out4, self.out4 * 2)
        self.fc_4_2 = nn.Linear(self.out4 * 2, self.out4)
        self.fc_4_3 = nn.Linear(self.out4, 16)

        self.out5 = 128
        self.conv5_1 = BasicConv(in_planes=self.out4,out_planes=self.out5)
        self.conv5_2 = BasicConv(in_planes=self.out5,out_planes=self.out5)
        self.dab5_2 = DAB(self.out5,kernel_size=3,reduction=8)



        self.conv6 = BasicConv(in_planes=self.out5,out_planes=self.out5)
        self.conv7 = BasicConv(in_planes=self.out5,out_planes=self.out5)

        # self.conv6 = BasicConv(in_planes=self.out4, out_planes=self.out4)
        # self.dab6 = DAB(self.out4, kernel_size=3, reduction=16)
        # self.out7 = 512
        # self.conv7 = BasicConv(in_planes=self.out4, out_planes=self.out7)
        # self.dab7 = DAB(self.out7, kernel_size=3, reduction=16)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc_all1 = nn.Linear(self.out5,self.out5*2)
        # self.fc2 = nn.Linear(self.out4,1)
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.fc_all2 = nn.Linear(self.out5*2, self.out5)
        self.fc_all3 = nn.Linear(self.out5, 16)

    def forward(self, inputs):
        bs = inputs.size(0)
        # inputs = dct.dct_2d(inputs)
        # print(inputs.size())
        # x = self.pool(inputs)
        x = self.conv1_1(inputs)
        # x = self.dab1(x)
        x1 = self.conv1_2(x)

        x = self.pool(x1)
        x = self.conv2_1(x)
        x2 = self.conv2_2(x)
        # x = self.dab2(x)
        out_2 = self._avg_pooling(x2)
        out_2 = out_2.view(bs, -1)
        out_2 = self.fc_2_1(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.fc_2_2(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.fc_2_3(out_2)
        out_2 = self.relu(out_2)



        x = self.pool(x2)
        x = self.conv3_1(x)
        x3 = self.conv3_2(x)
        # x = self.dab3(x)

        x = self.pool(x3)
        x = self.conv4_1(x)
        x4 = self.conv4_2(x)

        out_4 = self._avg_pooling(x4)
        out_4 = out_4.view(bs, -1)
        out_4 = self.fc_4_1(out_4)
        out_4 = self.relu(out_4)
        out_4 = self.fc_4_2(out_4)
        out_4 = self.relu(out_4)
        out_4 = self.fc_4_3(out_4)
        out_4 = self.relu(out_4)


        x = self.conv5_1(x4)
        x = self.conv5_2(x)
        x5 = self.dab5_2(x)

        x = self.conv6(x5)
        x = self.conv7(x)

        out = self._avg_pooling(x)
        out = out.view(bs, -1)
        out = self.fc_all1(out)
        # out = self.fc2(out)
        out = self.relu(out)
        out = self.fc_all2(out)
        out = self.relu(out)
        out = self.fc_all3(out)
        out = self.relu(out)
        return out, out_2, out_4
# import torch
if __name__ == "__main__":

    model = NormalModel(in_channel=3)
    # from torchsummary import summary_string
    import torchsummary
    import sys
    # print(sys.stdout)
    # sys.stdout = open("test.txt", "w")

    sum = torchsummary.summary(model,(3,224,224))
    # result, params_info = summary_string(model,(3,256,256),batch_size=-1, device=torch.device('cuda:0'), dtypes=None)
    # print(sum)
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__
    # print("aaa")

