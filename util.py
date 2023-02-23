import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torch
from torch.utils import data
from torch.nn import functional as F
from torch.autograd import Function
import random
import math
    
def visualize(img_arr):#visualize output
    plt.imshow(((img_arr.detach().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')

def save_image(img, filename):#save image
    tmp = ((img.detach().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
    
def load_image(filename):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    img = Image.open(filename)
    img = transform(img)
    return img.unsqueeze(dim=0)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

# train the adversarial network
def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

# calculate Lreg
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg     

def moving_average(model, model_test, beta=0.999):#EMA
    for param, param_test in zip(model.parameters(), model_test.parameters()):     
        param_test.data = torch.lerp(param.data, param_test.data, beta)

# Take the prediction of fake and real images from the combined batch
def divide_pred(pred):
    # the prediction contains the intermediate outputs of multiscale GAN
    # so it's usually a list
    if type(pred) == list:
        fake = []
        real = []
        for p in pred:
            fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
            real.append([tensor[tensor.size(0) // 2:] for tensor in p])
    else:
        fake = pred[:pred.size(0) // 2]
        real = pred[pred.size(0) // 2:]
    #print("fake_list="+str(fake)+"real_list="+str(real))
    return fake, real

