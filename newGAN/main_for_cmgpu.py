from prepare_data import data_loader, view_sample
from torch.utils.tensorboard import SummaryWriter
from networks import generatorX, generatorY, discriminator, vgg19
from loss_functions import generatorLoss, discriminatorLoss
import torch.optim as optim
from trainers import trainer
import torch


trial_name = "1222X/"
dataset_dir = "~/dl21_project/dataset/"
intermediate_results_path = "./progress/"+trial_name
checkpoints_path = "./checkpoints/"+trial_name
tb_log_dir = "./tensorboard/"+trial_name

batch_size = 24
image_size = 256
num_worker = 32
total_epoch = 1000

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

GX = generatorX().to(device=DEVICE)
GY = generatorY().to(device=DEVICE)
D = discriminator().to(device=DEVICE)

ext = vgg19().to(device=DEVICE)
G_Loss = generatorLoss(extractor=ext)
D_Loss = discriminatorLoss()

lr = 5e-5
lambda_gp = 10

GX_optim = optim.RMSprop(GX.parameters(), lr)
GY_optim = optim.RMSprop(GY.parameters(), lr)
D_optim = optim.RMSprop(D.parameters(), lr)

cartoonGAN_trainer = trainer(
    generatorX=GX, generatorY=GY, discriminator=D,
    generatorLoss=G_Loss, discriminatorLoss=D_Loss,
    photo_loader=photo_loader, cartoon_loader=cartoon_loader,
    GX_optim=GX_optim, GY_optim=GY_optim, D_optim=D_optim,
    lambda_gp=lambda_gp,
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
