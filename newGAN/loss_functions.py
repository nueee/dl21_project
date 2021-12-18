import torch.nn as nn
import torch


class generatorLoss(nn.Module):
    def __init__(self, w, vgg, device):
        super(generatorLoss, self).__init__()

        self.bce_loss = nn.BCELoss()
        self.l1loss = nn.L1Loss()
        self.feature_extractor = vgg.feature
        self.w = w
        self.device = device

    def forward(self, D_out_for_generated, G_in, G_out, current_epoch, image_size=256, tb_writer=None):
        content_loss = self._content_loss(G_in, G_out)
        if current_epoch < 10:
            adversarial_loss = 0.0
            g_loss = content_loss
        else:
            adversarial_loss = self._adversarial_loss_generator_part_only(D_out_for_generated, image_size)
            g_loss = adversarial_loss + self.w * content_loss

        if tb_writer:
            tb_writer.add_scalar('g_loss', g_loss, current_epoch)
            tb_writer.add_scalar('g_adversarial_loss', adversarial_loss, current_epoch)
            tb_writer.add_scalar('g_content_loss', content_loss, current_epoch)

        return g_loss

    def _adversarial_loss_generator_part_only(self, D_out_for_generated, image_size):
        actual_batch_size = D_out_for_generated.size()[0]
        target_ones = torch.ones([actual_batch_size, 1, image_size//4, image_size//4]).to(self.device)

        return self.bce_loss(D_out_for_generated, target_ones)

    def _content_loss(self, g_in, g_out):

        return self.l1loss(self.feature_extractor(g_out), self.feature_extractor(g_in))


class discriminatorLoss(nn.Module):
    def __init__(self, device):
        super(discriminatorLoss, self).__init__()

        self.bce_loss = nn.BCELoss()
        self.device = device

    def forward(self, D_out_for_generated, D_out_for_cartoon, D_out_for_smoothed, current_epoch, image_size=256, tb_writer=None):

        return self._adversarial_loss(D_out_for_generated, D_out_for_cartoon, D_out_for_smoothed, current_epoch, image_size, tb_writer)

    def _adversarial_loss(self, D_out_for_generated, D_out_for_cartoon, D_out_for_smoothed, current_epoch, image_size, tb_writer):
        actual_batch_size = D_out_for_generated.size()[0]
        target_zeros = torch.zeros([actual_batch_size, 1, image_size//4, image_size//4]).to(self.device)
        target_ones = torch.ones([actual_batch_size, 1, image_size//4, image_size//4]).to(self.device)

        adversarial_loss_G_input = self.bce_loss(D_out_for_generated, target_zeros)
        adversarial_loss_cartoon = self.bce_loss(D_out_for_cartoon, target_ones)
        adversarial_loss_smoothed = self.bce_loss(D_out_for_smoothed, target_zeros)

        d_loss = adversarial_loss_G_input + adversarial_loss_cartoon + adversarial_loss_smoothed

        if tb_writer:
            tb_writer.add_scalar('d_loss', d_loss, current_epoch)
            tb_writer.add_scalar('d_loss_generated', adversarial_loss_G_input, current_epoch)
            tb_writer.add_scalar('d_loss_cartoon', adversarial_loss_cartoon, current_epoch)
            tb_writer.add_scalar('d_loss_smoothed', adversarial_loss_smoothed, current_epoch)

        return d_loss


class newGeneratorLoss(nn.Module):
    def __init__(self):
        super(newGeneratorLoss, self).__init__()

    def forward(self, D_out_for_generated, current_epoch, tb_writer=None):
        g_loss = -torch.mean(D_out_for_generated)

        if tb_writer:
            tb_writer.add_scalar('g_loss', g_loss, current_epoch)

        return g_loss


class newDiscriminatorLoss(nn.Module):
    def __init__(self):
        super(newDiscriminatorLoss, self).__init__()

    def forward(self, D_out_for_generated, D_out_for_cartoon, current_epoch, tb_writer=None):
        d_loss = -torch.mean(D_out_for_cartoon) + torch.mean(D_out_for_generated)

        if tb_writer:
            tb_writer.add_scalar('d_loss', d_loss, current_epoch)

        return d_loss
