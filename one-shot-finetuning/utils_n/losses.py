from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self,device):
        super(CrossEntropyLoss, self).__init__()
        self.device = device
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        # print("cross entropy inputs size before view: %s" % (str(inputs.size())))
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        # print("cross entropy inputs size after view: %s" % (str(inputs.size())))
        log_probs = self.logsoftmax(inputs)
        # print("cross entropy function log_probs size after logsoftmax: %s" % (str(log_probs.size())))
        # print("cross entropy function log_probs after logsoftmax: %s" % (str(log_probs)))
        # print("cross entropy function targets after logsoftmax: %s" % (str(targets)))
        targets = torch.zeros(inputs.size(0), inputs.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.unsqueeze(-1)
        targets = targets.cuda(self.device)
        loss = (- targets * log_probs).mean(0).sum() 
        return loss / inputs.size(2)
