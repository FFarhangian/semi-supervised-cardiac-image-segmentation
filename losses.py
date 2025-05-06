import torch
from torch import nn
import torch.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    


class TverskyLoss(nn.Module):
    def __init__(self, alpha,beta,n_classes,weight=None, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_classes = n_classes
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def forward(self, inputs, targets,smooth=1):
        
        targets = self._one_hot_encoder(targets)

        #flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        #True Positives, False Positives & False Negatives
        TP = torch.sum(inputs * targets)   
        FP = torch.sum((1-targets) * inputs)
        FN = torch.sum(targets * (1-inputs))
    
        Tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  
        
        return 1 - Tversky
    


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, n_classes, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_classes = n_classes
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    

    def forward(self, inputs, targets, smooth=1e-5):
             
        targets = self._one_hot_encoder(targets)
        #flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()
        
        #True Positives, False Positives & False Negatives
        TP = torch.sum(inputs * targets)   
        FP = torch.sum((1-targets) * inputs)
        FN = torch.sum(targets * (1-inputs))
        
        Tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
                       
        return FocalTversky