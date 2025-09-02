import torch
import torch.nn as nn
import torch.nn.functional as F



## TO DO 1: For each of the modules given below complete the implementation
# using the figure and table given in the task pdf document.

class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2 times"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        # Step 3a
    
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1) #k = 3 always
        self.BN1 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(out_ch)
      
        

    def forward(self, x):

        # Step 3a
        
        x = self.relu(self.BN2((self.conv2(self.relu(self.BN1((self.conv1(x))))))))
        #x = self.conv(x)
        return x


class encoder(nn.Module):
    '''(maxpool => double_conv)'''

    def __init__(self, in_ch, out_ch):
        super(encoder, self).__init__()
        # Step 3b
        self.pool = nn.MaxPool2d(2)
        self.DC = double_conv(in_ch, out_ch)



    def forward(self, x):
        # Step 3b
        x = self.pool(x)
        x = self.DC.forward(x)
        
        return x


class decoder(nn.Module):
    """(up_conv x1 => concatenate with x2 => double_conv)
    x1: tensor output from previous layer (from below)
    x2: tensor output from encoder layer at same resolution level (from left)
    """

    def __init__(self, in_ch, out_ch):
        super(decoder, self).__init__()

        # Step 3c
        self.upconv    = nn.ConvTranspose2d(in_ch, out_ch, 2, 2) #2x2 transpose convolution as mentioned in table (HxW doubled; C halved)
        self.DC = double_conv(in_ch, out_ch)
        

    def forward(self, x1, x2):
        # Step 3c: Remove below line  x = None and complete the implementation
        #print("x1 shape ", x1.shape)
        x1 = self.upconv(x1)
       
        x = torch.cat((x2,x1), dim = 1)
        x = self.DC.forward(x)

        return x


class output_module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(output_module, self).__init__()
        # Step 3d
        self.conv1 = torch.nn.Conv1d(in_ch, out_ch, 2, padding=1, dilation=2) #k = 2 as given in question
        ###self.conv1 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        # Step 3d
        #print("before 1d conv", x.shape)
        x = x.view(x.shape[0], x.shape[1], (x.shape[2]*x.shape[3]))
        
        x = self.conv1(x)
        #print("after 1d conv", x.shape)
        x = x.view(x.shape[0], x.shape[1], 256, 256 )
        ###x = self.conv1(x)
        return x


## TO DO 2: Using the modules defined above, construct the complete U-Net
# architecture.

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        # Step 3e
        
        self.DC = double_conv(n_channels, 64)
        self.enc1 = encoder(64, 128)
        self.enc2 = encoder(128, 256)
        self.enc3 = encoder(256, 512)
        self.enc4 = encoder(512, 1024)
        
        self.dec1 = decoder(1024, 512)
        self.dec2 = decoder(512, 256)
        self.dec3 = decoder(256, 128)
        self.dec4 = decoder(128, 64)
        self.out = output_module(64, n_classes)
        

    def forward(self, x):
        # Step 3e
        #print("original", x.shape)
        x = self.DC.forward(x)
        #print("after first DC", x.shape)
        d4 = x
        x = self.enc1(x)
        #print("after enc 1", x.shape)
        d3 = x
        x = self.enc2(x)
        #print("after enc2 ", x.shape)
        d2 = x
        x = self.enc3(x)
        #print("after enc3 ", x.shape)
        d1 = x
        x = self.enc4(x)
        #print("after enc4 ", x.shape)
        x = self.dec1(x, d1)
        x = self.dec2(x, d2)
        x = self.dec3(x, d3)
        x = self.dec4(x, d4)
        x = self.out.forward(x)
    
        return x


## TO DO 3: Implement a network prediction function using the Pytorch
# softmax layer.

def get_network_prediction(network_output):
    """
    Using softmax on network output to get final prediction and prediction
    probability for the lane class.

    The input will have 4 dimension: N x C x H x W , where N: no of samples
    in mini-batch. This is defined as batch_size in the dataloader (see notebook).
    Recall from before C: Channels, H: Height, W: Width

    Both output tensors, i.e., predicted_labels and lane_probability will have
    3 dimensions: N x H X W

    """ 
    ## Step 3f: Delete the lines below and complete the implementation.
    #predicted_labels = 0
    #lane_probability = 0
    ##m = nn.Softmax(dim = 1)
    ##output = m(network_output) #output shape same as input i.e. N x C x H x W
    ##predicted_labels = output[:,1, :, :] > output[:,0,:,:] #shape N x H x W
 
    #predicted_labels = predicted_labels.astype('uint8')
    output = F.softmax(network_output,1)
    predicted_labels= torch.argmax(output,1)
    
    
    
    
   
    lane_probability = output[:, 1, :, :]
    
    ##predicted_labels = predicted_labels.int()
    #print(predicted_labels)
    #print(lane_probability)

    # Ensure that the probability tensor does not have the channel dimension






    return predicted_labels, lane_probability
