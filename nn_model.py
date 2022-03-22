class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4,stride=1,padding = 1)
        # self.conv1 = nn.Conv2d(2, 32, kernel_size=4,stride=1,padding = 1)
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=4,stride=1,padding = 1)
        self.mp1 = nn.MaxPool2d(kernel_size=3,stride=3)
        self.conv2 = nn.Conv2d(64,128, kernel_size=3,stride =1)
        self.mp2 = nn.MaxPool2d(kernel_size=5,stride=3)
        # self.fc1= nn.Linear(2304,256)
        # self.fc1= nn.Linear(5184,256)
        self.fc1= nn.Linear(2048,256)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256,3)

    def forward(self, x):
        in_size = x.size(0)
        # x = F.relu(self.mp1(self.conv1(x)))
        # x = F.relu(self.mp2(self.conv2(x)))
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = x.view(in_size,-1)
        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
