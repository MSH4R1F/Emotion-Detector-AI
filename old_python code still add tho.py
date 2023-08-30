import torch
# Importing torch - the machine learning framework
import torch.nn as nn

# This would apply convolutions to our images
import torch.nn.functional as F




class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        # This first part is similar to all Pytorch projects
        super(Deep_Emotion,self).__init__()
        # The super function makes class inheritance from nn.Module more manageable

        # Initializing our variables

        # We are using a conv2d layer as we are using a 2d image not a 3d image such as a video
        # Our first convolution has input channels: 1 which is a grayscale image
        # Our out channels is 10 as this is quite a large number allows our layers to learn more about the features of the input data.
        # Using out channel of 10 also uses less Ram as we need as my computer does not have a GPU.
        # Also this prevents overfitiing
        # kernel_size = 3. Kernel size is the size of the filters which is run over the images. Using a kernel size of 3 means it
        # looks at the features of this pizel and the pixels adjacent to it 
        self.conv1 = nn.Conv2d(1,10,3)

        # Our second convoltuon has input channels :10 which has more definition than an RGB image which has input channels 3
        self.conv2 = nn.Conv2d(10,10,3)




        #Applies a 2D max pooling over an input signal composed of several input planes.
        # This returns the maximum value of each 2x2 pixels 
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)


        # Batch normalization allows us the freedom of using larger learning rates without worryimg about internal probelsm
        # This allows us to run neural networks faster.
        self.batch_norm = nn.BatchNorm2d(10)


        # Tehse two variables apply a linear transformation t the dtaa 
        # Takes in input_features and output_features 
        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50,7)
        # Returns re all but the last dimension are the same shape as the input and


        # The sequenetial layer allows it to maek a neural network layer by layer

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)

        # Here we are reshaping our image dimensions


        theta = self.fc_loc(xs)


        theta = theta.view(-1, 2, 3) 

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,input):
        out = self.stn(input)   ## Our attention unit

        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.batch_norm(self.conv4(out))
        out = F.relu(self.pool4(out))

        out = F.dropout(out)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out