"this is the implementation of paper Resnet[Deep Residual Learning for Image Recognition]"


import torch  
from torch import nn, optim   
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class Basicblock(nn.Module):   
    
    expension = 1  

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):   

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias= False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):

        identity = x  
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)
    

class Resnet(nn.Module):  
      
     def __init__(self, block, layers, num_classes=1000):  
         
         super().__init__()
         self.in_channels = 64
         self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias= False)
         self.bn1 = nn.BatchNorm2d(64)
         self.pool = nn.MaxPool2d(3, 2, 1)

         self.layer1 = self._make_layer(block, 64,  layers[0])
         self.layer2 = self._make_layer(block, 128, layers[1], stride =2)
         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
         self.fc = nn.Linear(512 * block.expension, num_classes)  
         
         for m in self.modules():  
             if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')   
     
     def _make_layer(self, block, out_channels, blocks, stride=1):
         
         downsample = None 
         if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
         layers = [block(self.in_channels, out_channels, stride, downsample)]   
         self.in_channels = out_channels
         for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
         return nn.Sequential(*layers) 

     def forward(self, x): 
         
         x  = F.relu(self.bn1(self.conv1(x)))
         x = self.pool(x)
         x = self.layer1(x)
         x = self.layer2(x)
         x = self.layer3(x)
         x = self.layer4(x)
         x = self.avgpool(x)
         x = torch.flatten(x,1)
         return self.fc(x)
     
def resnet34(num_classes = 1000):
      return Resnet(Basicblock, [3, 4, 6, 3], num_classes) 


import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader  

transform = transforms.Compose([
    transforms.Resize(64),      
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)
testloader  = DataLoader(testset,  batch_size=16, shuffle=False, num_workers=0)


model = resnet34(num_classes=10).to(device)  
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  


if __name__ == '__main__': 
 for epoch in range(10):   

    model.train()
    running_loss = 0.0
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)   
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={running_loss/len(trainloader):.4f}")


 model.eval()
 correct = total = 0
 with torch.no_grad():
    for imgs, labels in testloader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
 print(f"Test accuracy: {100*correct/total:.2f}%")
