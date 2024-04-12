from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.simple_sigmoid_stack = nn.Sequential(
            nn.Linear(28*28, 30),
            nn.Sigmoid(),
            nn.Linear(30,10),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.simple_sigmoid_stack(x)
        return logits
    
