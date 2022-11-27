import torch
from torch.nn.functional import relu

"""
This file used codes from https://github.com/tobran/DF-GAN
"""


def hinge_loss(output, target):
    if target == 0:
        err = relu(1.0 - output).mean()
    else:
        err = relu(1.0 + output).mean()
    return err


def MA_GP(img, sent, out):
    grads = torch.autograd.grad(outputs=out,
                                inputs=(img, sent),
                                grad_outputs=torch.ones(out.size()).cuda(),
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0, grad1), dim=1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp = 2.0 * torch.mean(grad_l2norm ** 6)
    return d_loss_gp
