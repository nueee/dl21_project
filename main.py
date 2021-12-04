from prepare_data import data_loader, view_sample
from torch.utils.tensorboard import SummaryWriter
from neural_nets import generator, discriminator, vgg16, vgg19
from loss_functions import generatorLoss, discriminatorLoss
import torch


dataset_dir = "dataset/"
tb_log_dir = "tensorboard"

batch_size = 16
image_size = 256
num_worker = 12
num_epoch = 200
DEVICE = None

cartoon_loader = data_loader(
    image_dir=dataset_dir+"cartoons",
    batch_size=batch_size,
    image_size=image_size,
    num_workers=num_worker
)
photo_loader = data_loader(
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

G_Loss = generatorLoss(w=10.0, vgg=VGG, device=DEVICE)
D_Loss = discriminatorLoss(device=DEVICE)


