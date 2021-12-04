from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import math

train_ratio = 0.9


def data_loader(image_dir, batch_size, image_size, num_workers):
    tf = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    image_data = ImageFolder(image_dir, transform=tf)
    num_train = math.floor(len(image_data) * train_ratio)
    num_test = len(image_data) - num_train
    train_set, test_set = random_split(image_data, (num_train, num_test))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_loader, test_loader


def view_sample(image_loader):
    iterator = iter(image_loader)
    sample_batch, _ = iterator
    first_sample_image_of_batch = sample_batch[0]
    print(first_sample_image_of_batch.size())
    print("Current range: {} to {}".format(first_sample_image_of_batch.min(), first_sample_image_of_batch.max()))
    plt.imshow(np.transpose(first_sample_image_of_batch.numpy(), (1, 2, 0)))
