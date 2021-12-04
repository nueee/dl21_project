from prepare_data import data_loader, view_sample
from torch.utils.tensorboard import SummaryWriter
from neural_nets import generator, discriminator, vgg16, vgg19
from loss_functions import generatorLoss, discriminatorLoss
from trainers import trainer
import torch
import numpy as np
import matplotlib.pyplot as plt


trial_name = "1204A/"
dataset_dir = "dataset/"
intermediate_results_path = "intermediate_results/"+trial_name
checkpoints_path = "checkpoints/"+trial_name
tb_log_dir = "tensorboard/"+trial_name

batch_size = 16
image_size = 256
num_worker = 12
total_epoch = 200
gen_loss_w = 10.0
DEVICE = None

cartoon_loader = data_loader(
    image_dir=dataset_dir+"cartoons",
    batch_size=batch_size,
    image_size=image_size,
    num_workers=num_worker
)
photo_loader, photo_test_loader = data_loader(
    image_dir=dataset_dir+"cartoons",
    batch_size=batch_size,
    image_size=image_size,
    num_workers=num_worker
)
smoothed_loader = data_loader(
    image_dir=dataset_dir+"cartoons",
    batch_size=batch_size,
    image_size=image_size,
    num_workers=num_worker
)

view_sample(cartoon_loader)
view_sample(photo_loader)
view_sample(smoothed_loader)

tb_writer = SummaryWriter(tb_log_dir)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Train on GPU.")
else:
    DEVICE = torch.device('cpu')
    print("No cuda available.\nTrain on CPU.")

G = generator().to(DEVICE)
D = discriminator().to(DEVICE)
VGG = vgg19().to(DEVICE)

G_Loss = generatorLoss(w=gen_loss_w, vgg=VGG, device=DEVICE)
D_Loss = discriminatorLoss(device=DEVICE)

cartoonGAN_trainer = trainer(
    generator=G, discriminator=D,
    generatorLoss=G_Loss, discriminatorLoss=D_Loss,
    photo_loader=photo_loader, cartoon_loader=cartoon_loader, smoothed_loader=smoothed_loader,
    image_size=image_size,
    device=DEVICE
)

checkpoint_to_load = ""
if checkpoint_to_load == "":
    print("start training from scratch")
else:
    cartoonGAN_trainer.load_checkpoint(checkpoint_to_load)
    print("continue training from loaded model")

losses = cartoonGAN_trainer.train(
    total_epoch=total_epoch,
    image_path=intermediate_results_path,
    checkpoint_path=checkpoints_path,
    tb_writer=tb_writer
)

d_losses = [x[0] for x in losses]
g_losses = [x[1] for x in losses]
g_adversarial_loss = [x[2][3] for x in losses]
g_content_loss = [x[2][4] for x in losses]

fig1 = plt.figure()
plt.plot(d_losses, label='Discriminator training loss')
plt.plot(g_losses, label='Generator training loss')
plt.legend(frameon=False)

fig2 = plt.figure()
plt.plot(g_adversarial_loss, label='Generator adversarial loss')
plt.plot(g_content_loss, label='Generator content loss')
plt.legend(frameon=False)

# test
target_epoch = input("epoch to test : ")
checkpoint_to_test = checkpoints_path+"checkpoint_epoch_0"+target_epoch+".pth"
model_to_test = torch.load(checkpoint_to_test)

G_inference = generator().to(DEVICE)
G_inference.load_state_dict(model_to_test["G_state_dict"])
D_inference = discriminator().to(DEVICE)
D_inference.load_state_dict(model_to_test["D_state_dict"])

for index, (photos, _) in enumerate(photo_test_loader):
    photos = photos.to(DEVICE)

    with torch.no_grad():
        G_inference.eval()
        D_inference.eval()
        g_test_loss = G_Loss(
            D_inference(G_inference(photos)),
            photos,  G_inference(photos), target_epoch, image_size
        )
        print("g_test_loss {:6.4f}".format(g_test_loss.item()))

    if index % 10 == 0:
        image_input = photos[0].detach().cpu().numpy()
        image_output = G_inference(photos)[0].detach().cpu().numpy()
        image_input = np.transpose(image_input, (1, 2, 0))
        image_output = np.transpose(image_output, (1, 2, 0))

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        plt.imshow(image_input)
        ax2 = fig.add_subplot(1, 2, 2)
        plt.imshow(image_output)
