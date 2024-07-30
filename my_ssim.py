import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

import cv2
import numpy as np
# from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
import torchvision

class SSIMLossWithThreshold(torch.nn.Module):
    def __init__(self, threshold=0.05, window_size=11, reduction='mean'):
        super(SSIMLossWithThreshold, self).__init__()
        self.threshold = threshold
        self.window_size = window_size
        self.reduction = reduction

    def forward(self, img1, img2):
        ssim_index = ssim(img1, img2, self.window_size)
        ssim_loss = 1 - ssim_index  # SSIM loss

        # Compute pixel-wise absolute difference
        abs_diff = torch.abs(img1 - img2)

        # Apply threshold: if difference is less than threshold, set loss to zero for those pixels
        threshold_mask = (abs_diff < self.threshold).float()
        weighted_ssim_loss = ssim_loss * threshold_mask

        # Reduce loss (mean or sum)
        if self.reduction == 'mean':
            return torch.mean(weighted_ssim_loss)
        elif self.reduction == 'sum':
            return torch.sum(weighted_ssim_loss)
        else:
            return weighted_ssim_loss

def color_preservation_loss(input_image, generated_image):
    # Compute SSIM loss
    ssim_loss = 1 - ssim(input_image, generated_image)
    abs_diff = torch.abs(input_image - generated_image)
    total_diff = abs_diff.sum()
    total_loss = ssim_loss + total_diff*0.00001
        
    return total_loss

