import matplotlib.pyplot as plt
import numpy as np
import time
from torch import save, load, no_grad


def save_training_image_result(inputs, outputs, current_epoch, path):
    image_input = inputs[0].detach().cpu().numpy()
    image_output = outputs[0].detach().cpu().numpy()
    image_input = np.transpose(image_input, (1, 2, 0))
    image_output = np.transpose(image_output, (1, 2, 0))

    filename = str(current_epoch)
    path_input = path + filename + "_input.jpg"
    path_output = path + filename + "_output.jpg"
    plt.imsave(path_input, image_input)
    plt.imsave(path_output, image_output)


class newTrainer:
    def __init__(
        self,
        generator, discriminator,
        generatorLoss, discriminatorLoss,
        photo_loader, cartoon_loader,
        G_optim, D_optim,
        weight_clip_range,
        device
    ):
        self.device = device

        self.G = generator.to(device)
        self.D = discriminator.to(device)

        self.G_Loss = generatorLoss
        self.D_Loss = discriminatorLoss

        self.photo_loader = photo_loader
        self.cartoon_loader = cartoon_loader

        self.current_epoch = 0

        self.G_optim = G_optim
        self.D_optim = D_optim

        self.weight_clip_range = weight_clip_range

    def train(self, total_epoch, image_path, checkpoint_path, tb_writer=None):
        for epoch in range(total_epoch):
            self.current_epoch = epoch
            prev_time = time.time()

            photos = None
            G_photos = None

            for index, ((photos, _), (cartoons, _)) in enumerate(
                    zip(self.photo_loader, self.cartoon_loader)
            ):
                photos = photos.to(self.device)
                cartoons = cartoons.to(self.device)

                self.D.train()
                self.G.train()

                # discriminator
                self.D_optim.zero_grad()

                D_G_photos = self.D(self.G(photos))
                D_cartoons = self.D(cartoons)

                d_loss = self.D_Loss(D_G_photos, D_cartoons, self.current_epoch, tb_writer)
                d_loss.backward()
                self.D_optim.step()

                for p in self.D.parameters():
                    p.data.clamp_(-self.weight_clip_range, self.weight_clip_range)

                # generator
                self.G_optim.zero_grad()

                G_photos = self.G(photos)
                D_G_photos = self.D(G_photos)

                g_loss = self.G_Loss(D_G_photos, self.current_epoch, tb_writer)
                g_loss.backward()
                self.G_optim.step()

                if index % 50 == 0:
                    curr_time = time.time()
                    elapsed_time = curr_time - prev_time
                    print(
                        "Epoch {}/{} | d_loss {:6.4f} | g_loss {:6.4f} | time {:2.0f}s".format(
                            epoch+1, total_epoch, d_loss.item(), g_loss.item(), elapsed_time
                        )
                    )
                    prev_time = curr_time

            save_training_image_result(photos, G_photos, epoch, image_path)
            self.save_checkpoint(checkpoint_path + '/checkpoint_epoch_{:03d}.pth'.format(epoch + 1))

    def save_checkpoint(self, path):
        print("Save checkpoint for epoch {}".format(self.current_epoch + 1))
        save({
            'current_epoch': self.current_epoch,
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'G_optim_state_dict': self.G_optim.state_dict(),
            'D_optim_state_dict': self.D_optim.state_dict()
        }, path)

    def load_checkpoint(self, path):
        checkpoint = load(path)
        self.current_epoch = checkpoint['current_epoch']
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])
        self.G_optim.load_state_dict(checkpoint['G_optim_state_dict'])
        self.D_optim.load_state_dict(checkpoint['D_optim_state_dict'])
