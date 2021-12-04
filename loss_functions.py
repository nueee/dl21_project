import torch.nn as nn
from torch import ones, zeros


class generatorLoss(nn.Module):
    def __init__(self, w, vgg, device):
        super(generatorLoss, self).__init__()

        self.bce_loss = nn.BCELoss()
        self.feature_extractor = vgg.features

        self.w = w
        self.device = device

        for param in self.feature_extractor.parameters():
            param.require_grad = False

    def forward(self, D_out_for_generated, G_in, G_out, current_epoch, image_size=256, tb_writer=None):
        if current_epoch < 10:
            g_content_loss = self._content_loss(G_in, G_out)
            g_adversarial_loss = 0.0
            g_loss = g_content_loss
        else:
            g_adversarial_loss = self._adversarial_loss_generator_part_only(D_out_for_generated, image_size)
            g_content_loss = self._content_loss(G_in, G_out)
            g_loss = g_adversarial_loss + self.w * g_content_loss

        if tb_writer:
            tb_writer.add_scalar('g_adversarial_loss', g_adversarial_loss, current_epoch)
            tb_writer.add_scalar('g_content_loss', g_content_loss, current_epoch)
            tb_writer.add_scalar('g_loss', g_loss, current_epoch)

        return g_loss

    def _adversarial_loss_generator_part_only(self, D_out_for_generated, image_size):
        actual_batch_size = D_out_for_generated.size()[0]
        target_ones = ones([actual_batch_size, 1, image_size//4, image_size//4]).to(self.device)
        return self.bce_loss(D_out_for_generated, target_ones)

    def _content_loss(self, g_in, g_out):
        return (self.feature_extractor(g_out) - self.feature_extractor(g_in)).norm(p=1)


class discriminatorLoss(nn.Module):
    def __init__(self, device):
        super(discriminatorLoss, self).__init__()

        self.bce_loss = nn.BCELoss()
        self.device = device

    def forward(self, D_out_for_cartoon, D_out_for_smoothed, D_out_for_generated, current_epoch, image_size=256, tb_writer=None):

        return self._adversarial_loss(D_out_for_cartoon, D_out_for_smoothed, D_out_for_generated, current_epoch, image_size, tb_writer)

    def _adversarial_loss(self, D_out_for_cartoon, D_out_for_smoothed, D_out_for_generated, current_epoch, image_size, tb_writer):
        actual_batch_size = D_out_for_cartoon.size()[0]
        target_zeros = zeros([actual_batch_size, 1, image_size//4, image_size//4]).to(self.device)
        target_ones = ones([actual_batch_size, 1, image_size//4, image_size//4]).to(self.device)

        d_loss_cartoon = self.bce_loss(D_out_for_cartoon, target_ones)
        d_loss_smoothed = self.bce_loss(D_out_for_smoothed, target_zeros)
        d_loss_generated = self.bce_loss(D_out_for_generated, target_zeros)

        d_loss = d_loss_cartoon + d_loss_smoothed + d_loss_generated

        if tb_writer:
            tb_writer.add_scalar('d_loss_cartoon', d_loss_cartoon, current_epoch)
            tb_writer.add_scalar('d_loss_smoothed', d_loss_smoothed, current_epoch)
            tb_writer.add_scalar('d_loss_generated', d_loss_generated, current_epoch)
            tb_writer.add_scalar('d_loss', d_loss, current_epoch)

        return d_loss
