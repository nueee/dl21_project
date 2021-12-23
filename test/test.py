import torch
from loss_functions import generatorLoss, discriminatorLoss
from neural_nets import generator, discriminator

# not completed code!!!

checkpoint_to_test = "checkpoints/checkpoint_epoch_"
model_to_test = torch.load(checkpoint_to_test)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Train on GPU.")
else:
    DEVICE = torch.device('cpu')
    print("No cuda available.\nTrain on CPU.")

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
            photos,  G_inference(photos), int(target_epoch), image_size
        )
        print("g_test_loss {:6.4f}".format(g_test_loss.item()))

    if index % 10 == 0:
        image_input = photos[0].detach().cpu().numpy()
        image_output = G_inference(photos)[0].detach().cpu().numpy()
        image_input = np.transpose(image_input, (1, 2, 0))
        image_output = np.transpose(image_output, (1, 2, 0))

