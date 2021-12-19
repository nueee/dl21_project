import torch.nn as nn
import torch


class generatorLoss(nn.Module):
    def __init__(self):
        super(generatorLoss, self).__init__()

    def forward(self, D_out_for_generated, current_epoch, tb_writer=None):
        g_loss = -torch.mean(D_out_for_generated)  # generator portion

        if tb_writer:
            tb_writer.add_scalar('g_loss', g_loss, current_epoch)

        return g_loss


class discriminatorLoss(nn.Module):
    def __init__(self):
        super(discriminatorLoss, self).__init__()

    def forward(self, D_out_for_generated, D_out_for_cartoon, current_epoch, tb_writer=None):
        d_loss = -torch.mean(D_out_for_cartoon) + torch.mean(D_out_for_generated)  # negative because of argmax, SGA

        if tb_writer:
            tb_writer.add_scalar('d_loss', d_loss, current_epoch)

        return d_loss
