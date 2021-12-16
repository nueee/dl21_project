from prepare_data import data_loader, view_sample
from torch.utils.tensorboard import SummaryWriter
from neural_nets import generator, discriminator, vgg16, vgg19, newDiscriminator
from loss_functions import generatorLoss, discriminatorLoss, newGeneratorLoss, newDiscriminatorLoss
import torch.optim as optim
from trainers import trainer, newTrainer
import torch


trial_name = "1216D/"
dataset_dir = "dataset/"
intermediate_results_path = "intermediate_results/"+trial_name
checkpoints_path = "checkpoints/"+trial_name
tb_log_dir = "tensorboard/"+trial_name

batch_size = 16
image_size = 256
num_worker = 24
total_epoch = 200
weight_clip_range = 0.1

cartoon_loader, _ = data_loader(
    image_dir=dataset_dir+"cartoons",
    batch_size=batch_size,
    image_size=image_size,
    num_workers=num_worker
)
photo_loader, photo_test_loader = data_loader(
    image_dir=dataset_dir+"photos",
    batch_size=batch_size,
    image_size=image_size,
    num_workers=num_worker
)

view_sample(cartoon_loader)
view_sample(photo_loader)

tb_writer = SummaryWriter(tb_log_dir)

DEVICE = None
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Train on GPU.")
else:
    DEVICE = torch.device('cpu')
    print("No cuda available.\nTrain on CPU.")

G = generator().to(DEVICE)
D = newDiscriminator().to(DEVICE)

G_Loss = newGeneratorLoss()
D_Loss = newDiscriminatorLoss()

lr = 3e-5

G_optim = optim.RMSprop(G.parameters(), lr)
D_optim = optim.RMSprop(D.parameters(), lr)

cartoonGAN_trainer = newTrainer(
    generator=G, discriminator=D,
    generatorLoss=G_Loss, discriminatorLoss=D_Loss,
    photo_loader=photo_loader, cartoon_loader=cartoon_loader,
    G_optim=G_optim, D_optim=D_optim,
    weight_clip_range=weight_clip_range,
    device=DEVICE
)

checkpoint_to_load = ""
if checkpoint_to_load == "":
    print("start training from scratch")
else:
    cartoonGAN_trainer.load_checkpoint(checkpoint_to_load)
    print("continue training from loaded model")

cartoonGAN_trainer.train(
    total_epoch=total_epoch,
    image_path=intermediate_results_path,
    checkpoint_path=checkpoints_path,
    tb_writer=tb_writer
)
