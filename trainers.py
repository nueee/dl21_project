import torch.optim as opt
import matplotlib.pyplot as plt
import numpy as np
import time
from torch import save, load


def save_training_image_result(inputs, outputs, current_epoch, path):
    # check 를 위하여 input, output image 하나씩 저장
    image_input = inputs[0].detach().cpu().numpy()
    image_output = outputs[0].detach().cpu().numpy()
    image_input = np.transpose(image_input, (1, 2, 0))
    image_output = np.transpose(image_output, (1, 2, 0))

    # 현재 epoch 를 해당 image 의 이름으로 저장
    filename = str(current_epoch)
    path_input = path + filename + "_input.jpg"
    path_output = path + filename + "_output.jpg"
    plt.imsave(path_input, image_input)
    plt.imsave(path_output, image_output)

    # 위의 image 출력
    # plt.imshow(image_input)
    # plt.imshow(image_output)


class trainer:
    def __init__(
            self,
            generator, discriminator,
            generatorLoss, discriminatorLoss,
            photo_loader, cartoon_loader, smoothed_loader,
            batch_size, image_size,
            device
    ):
        self.device = device

        self.G = generator.to(device)
        self.D = discriminator.to(device)

        self.G_Loss = generatorLoss
        self.D_Loss = discriminatorLoss

        self.photo_loader = photo_loader
        self.cartoon_loader = cartoon_loader
        self.smoothed_loader = smoothed_loader

        self.batch_size = batch_size
        self.image_size = image_size

        self.current_epoch = 0

        # optimizer 정의, learning rate, beta 값 수정 가능, optimizer 종류도 수정 가능
        self.lr = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.G_opt = opt.Adam(self.G.parameters(), self.lr, (self.beta1, self.beta2))
        self.D_opt = opt.Adam(self.D.parameters(), self.lr, (self.beta1, self.beta2))

        # loss history 저장
        self.losses = []

    def train(self, total_epoch, image_path, checkpoint_path):

        for epoch in range(total_epoch):
            self.current_epoch = epoch
            start_time = time.time()
            photos = None
            G_photos = None

            for index, ((photos, _), (cartoons, _), (smoothed, _)) in enumerate(
                    zip(self.photo_loader, self.cartoon_loader, self.smoothed_loader)):

                photos = photos.to(self.device)
                cartoons = cartoons.to(self.device)
                smoothed = smoothed.to(self.device)

                self.D.train()
                self.G.train()

                # discriminator 학습
                self.D_opt.zero_grad()

                # loss 계산을 위한 값들 계산
                D_G_photos = self.D(self.G(photos))
                D_cartoons = self.D(cartoons)
                D_smoothed = self.D(smoothed)

                # loss 계산
                d_loss = self.D_Loss(D_G_photos, D_cartoons, D_smoothed, self.batch_size, self.image_size)

                d_loss.backward()
                self.D_opt.step()

                # generator 학습
                self.G_opt.zero_grad()

                # loss 계산을 위한 값들 계산
                G_photos = self.G(photos)
                D_G_photos = self.D(G_photos)

                # loss 계산
                g_loss = self.G_Loss(D_G_photos, photos, G_photos, self.current_epoch, self.batch_size, self.image_size)

                g_loss.backward()
                self.G_opt.step()

                # 100번 마다 loss 값과 시간 출력
                if index % 50 == 0:
                    # 혹시 모를 분석을 위해 세부 loss 도 저장
                    extra_losses = (
                        self.D_Loss.adversarial_loss_G_input,
                        self.D_Loss.adversarial_loss_cartoon,
                        self.D_Loss.adversarial_loss_smoothed,
                        self.G_Loss.adversarial_loss,
                        self.G_Loss.content_loss
                    )

                    self.losses.append((d_loss.item(), g_loss.item(), extra_losses))
                    now = time.time()
                    current_run_time = now - start_time
                    start_time = now
                    print(
                        "Epoch {}/{} | d_loss {:6.4f} | g_loss {:6.4f} | time {:2.0f}s | total no. of losses {}".format(
                            epoch + 1, total_epoch, d_loss.item(), g_loss.item(), current_run_time, len(self.losses)
                        )
                    )

            # 각 epoch 마다 중간 결과 이미지 저장, 체크포인트 저장
            save_training_image_result(photos, G_photos, epoch, image_path)
            self.save_checkpoint(checkpoint_path + '/checkpoint_epoch_{:03d}.pth'.format(epoch + 1))

        return self.losses

    def save_checkpoint(self, path):
        # 중간에 끊길 것을 대비하여 check point 저장
        print("Save checkpoint for epoch {}".format(self.current_epoch + 1))
        save({
            'current_epoch': self.current_epoch,
            'losses': self.losses,
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'G_optim_state_dict': self.G_opt.state_dict(),
            'D_optim_state_dict': self.D_opt.state_dict()
        }, path)

    def load_checkpoint(self, path):
        # 해당 path 에 있는 checkpoint 를 load
        checkpoint = load(path)
        self.losses = checkpoint['losses']
        self.current_epoch = checkpoint['current_epoch']
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])
        self.G_opt.load_state_dict(checkpoint['G_optim_state_dict'])
        self.D_opt.load_state_dict(checkpoint['D_optim_state_dict'])
