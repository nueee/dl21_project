import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import save, load, no_grad


class trainer:
    def __init__(
        self,
        generatorX, generatorY, discriminator,
        generatorLoss, discriminatorLoss,
        photo_loader, cartoon_loader,
        GX_optim, GY_optim, D_optim,
        lambda_gp,
        device
    ):
        self.device = device

        self.GX = generatorX.to(device)
        self.GY = generatorY.to(device)
        self.D = discriminator.to(device)

        self.G_Loss = generatorLoss
        self.D_Loss = discriminatorLoss

        self.photo_loader = photo_loader
        self.cartoon_loader = cartoon_loader

        self.current_epoch = 0

        self.GX_optim = GX_optim
        self.GY_optim = GY_optim
        self.D_optim = D_optim

        self.lambda_gp = lambda_gp

        self.photos = None
        self.cartoons = None
        self.generated = None

    def train(self, total_epoch, image_path, checkpoint_path, tb_writer=None):
        for epoch in range(self.current_epoch, total_epoch):
            self.current_epoch = epoch
            prev_time = time.time()

            d_loss = None
            g_loss = None

            for index, ((photos, _), (cartoons, _)) in enumerate(
                    zip(self.photo_loader, self.cartoon_loader)
            ):
                self.photos = photos.to(self.device)
                self.cartoons = cartoons.to(self.device)

                self.GX.train()
                self.GY.train()
                self.D.train()

                if self.current_epoch < 30:
                    x = torch.rand((self.cartoons.shape[0], 512, 32, 32)).to(self.device)
                else:
                    x = self.GX(self.photos)

                y = torch.rand((self.cartoons.shape[0], 512, 32, 32)).to(self.device)

                # ------------------------------------ discriminator
                self.D_optim.zero_grad()

                self.generated = self.GY(x, y)

                D_G_photos = self.D(self.generated)
                D_cartoons = self.D(self.cartoons)

                gradient_penalty = self.compute_gradient_penalty()

                d_loss = self.D_Loss(D_G_photos, D_cartoons, self.current_epoch, tb_writer)
                d_loss += (self.lambda_gp + gradient_penalty)

                d_loss.backward()

                self.D_optim.step()

                # ------------------------------------ generator
                self.GX_optim.zero_grad()
                self.GY_optim.zero_grad()

                if self.current_epoch >= 30:
                    x = self.GX(self.photos)

                self.generated = self.GY(x, y)

                D_G_photos = self.D(self.generated)

                g_loss = self.G_Loss(D_G_photos, self.generated, self.photos, self.current_epoch, tb_writer)

                g_loss.backward()

                if self.current_epoch < 30:
                    pass
                else:
                    self.GX_optim.step()

                self.GY_optim.step()

            elapsed_time = time.time() - prev_time
            print(
                "Epoch {}/{} | d_loss {:6.4f} | g_loss {:6.4f} | time {:2.0f}s".format(
                    self.current_epoch+1, total_epoch, d_loss.item(), g_loss.item(), elapsed_time
                )
            )

            self.save_training_image_result(image_path)
            self.save_checkpoint(checkpoint_path + '/checkpoint_epoch_{:03d}.pth'.format(self.current_epoch + 1))

    def compute_gradient_penalty(self):
        alpha = torch.cuda.FloatTensor(np.random.random((self.cartoons.size(0), 1, 1, 1)))
        interpolates = (alpha * self.cartoons + ((1 - alpha) * self.generated)).requires_grad_(True)
        d_interpolates = self.D(interpolates)
        fake = torch.autograd.Variable(torch.cuda.FloatTensor(self.cartoons.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
        # print(d_interpolates.shape, interpolates.shape, fake.shape)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def save_training_image_result(self, path):
        image_photo = self.photos[0].detach().cpu().numpy()
        image_cartoon = self.cartoons[0].detach().cpu().numpy()
        image_output = self.generated[0].detach().cpu().numpy()
        image_photo = np.transpose(image_photo, (1, 2, 0))
        image_cartoon = np.transpose(image_cartoon, (1, 2, 0))
        image_output = np.transpose(image_output, (1, 2, 0))

        filename = str(self.current_epoch)
        path_photo = path + filename + "_photo.jpg"
        path_cartoon = path + filename + "_cartoon.jpg"
        path_output = path + filename + "_output.jpg"
        plt.imsave(path_photo, image_photo)
        plt.imsave(path_cartoon, image_cartoon)
        plt.imsave(path_output, image_output)

    def save_checkpoint(self, path):
        print("Save checkpoint for epoch {}".format(self.current_epoch + 1))
        save({
            'current_epoch': self.current_epoch,
            'GX_state_dict': self.GX.state_dict(),
            'GY_state_dict': self.GY.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'GX_optim_state_dict': self.GX_optim.state_dict(),
            'GY_optim_state_dict': self.GY_optim.state_dict(),
            'D_optim_state_dict': self.D_optim.state_dict()
        }, path)

    def load_checkpoint(self, path):
        checkpoint = load(path)
        self.current_epoch = checkpoint['current_epoch']
        self.GX.load_state_dict(checkpoint['GX_state_dict'])
        self.GY.load_state_dict(checkpoint['GY_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])
        self.GX_optim.load_state_dict(checkpoint['GX_optim_state_dict'])
        self.GY_optim.load_state_dict(checkpoint['GY_optim_state_dict'])
        self.D_optim.load_state_dict(checkpoint['D_optim_state_dict'])
