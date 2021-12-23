import torch.nn as nn
import torch


class generatorLoss(nn.Module):
    def __init__(self, extractor):
        super(generatorLoss, self).__init__()

        self.l1loss = nn.L1Loss()
        self.ext = extractor.feature

    def forward(self, D_out_for_generated, generated, reference, current_epoch, tb_writer=None):
        adversarial_loss = -torch.mean(D_out_for_generated)  # generator portion

        if current_epoch < 10:
            content_loss = 0.0
        else:
            content_loss = self.l1loss(self.ext(generated), self.ext(reference))

        g_loss = 1e-2*adversarial_loss + content_loss

        if tb_writer:
            tb_writer.add_scalar('generator/total_loss', g_loss, current_epoch)
            tb_writer.add_scalar('generator/content_loss', content_loss, current_epoch)
            tb_writer.add_scalar('generator/adversarial_loss', adversarial_loss, current_epoch)

        return g_loss


class discriminatorLoss(nn.Module):
    def __init__(self):
        super(discriminatorLoss, self).__init__()

    def forward(self, D_out_for_generated, D_out_for_cartoon, current_epoch, tb_writer=None):
        adversarial_loss = -torch.mean(D_out_for_cartoon) + torch.mean(D_out_for_generated)

        d_loss = 1e-2*adversarial_loss

        if tb_writer:
            tb_writer.add_scalar('discriminator/total_loss', d_loss, current_epoch)
            tb_writer.add_scalar('discriminator/adversarial_loss', adversarial_loss, current_epoch)

        return d_loss
