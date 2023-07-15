from torch import nn
from torchinfo import summary
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self,dof):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(dof, 256, 15, padding=7, padding_mode='replicate'),
                                     nn.ReLU(),
                                     nn.MaxPool1d(2),
                                     nn.Dropout(0.2),

                                     nn.Conv1d(256, 128, 15, padding=7, padding_mode='replicate'),
                                     nn.ReLU(),
                                     nn.MaxPool1d(2),
                                     nn.Dropout(0.2),


                                     # nn.Conv1d(10, 2, 15, padding=7, padding_mode='replicate'),
                                     # # nn.MaxPool1d(2),
                                     # nn.Tanh()
                                     
                                     )
        self.MLP = nn.Sequential(nn.Linear(128*7,128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 
                                 nn.Linear(128,32),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),

                                 nn.Linear(32,20),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),

                                )
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1,128*7)
        x = self.MLP(x)      
        return F.log_softmax(x, dim=1)
    



encoder = AutoEncoder(54)
batch_size = 8
summary(model=encoder, input_size=(batch_size, 54, 30))
