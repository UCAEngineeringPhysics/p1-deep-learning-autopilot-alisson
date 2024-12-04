import torch.nn as nn
import torch.nn.functional as F

class DonkeyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.fc1 = nn.Linear(64*8*13, 128)  # (64*30*30, 128) for 300x300 images
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):               #   300x300                     #  120x160
        x = self.relu(self.conv24(x))  # (300-5)/2+1 = 148     |     (120-5)/2+1 = 58   (160-5)/2+1 = 78
        x = self.relu(self.conv32(x))  # (148-5)/2+1 = 72      |     (58 -5)/2+1 = 27   (78 -5)/2+1 = 37
        x = self.relu(self.conv64_5(x))  # (72-5)/2+1 = 34     |     (27 -5)/2+1 = 12   (37 -5)/2+1 = 17
        x = self.relu(self.conv64_3(x))  # 34-3+1 = 32         |     12 - 3 + 1  = 10   17 - 3 + 1  = 15
        x = self.relu(self.conv64_3(x))  # 32-3+1 = 30         |     10 - 3 + 1  = 8    15 - 3 + 1  = 13

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


### START CODING HERE ###
class AutopilotNet(nn.Module):

    def __init__(self):
        super().__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))  # (176x208) -> (86x102)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))  # (86x102) -> (41x49)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))  # (41x49) -> (19x23)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))  # (19x23) -> (17x21)
        
        # Fully connected layers
        # The size after convolutions: (64, 17, 21), so we need to calculate the number of input features for the first FC layer
        self.fc1 = nn.Linear(64 * 17 * 21, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # Output steering and throttle

        # Activation function
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Apply the convolutional layers with ReLU activation
        x = self.relu(self.conv1(x))  # Output size: (batch_size, 24, 86, 102)
        x = self.relu(self.conv2(x))  # Output size: (batch_size, 32, 41, 49)
        x = self.relu(self.conv3(x))  # Output size: (batch_size, 64, 18, 22)
        x = self.relu(self.conv4(x))  # Output size: (batch_size, 64, 16, 20)

        # Flatten the output from the conv layers
        x = self.flatten(x)  # Flatten to (batch_size, 64 * 16 * 20)

        # Apply fully connected layers with ReLU activations
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Output steering and throttle values
        x = self.fc3(x)  # Final output of shape (batch_size, 2)

        return x
### END CODING HERE ###


