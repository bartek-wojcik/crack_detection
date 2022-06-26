import os
import numpy as np
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import transforms
from autoencoder import Autoencoder
from dataset import CrackDataset
from utils import visualize_reconstructions, get_images, visualize_confusion_matrix

transform = transforms.Compose([transforms.ToTensor()])

# Loading the training dataset. We need to split it into a training and validation part
train_dataset = CrackDataset(root='train', transform=transform)
seed_everything(42)
train_set, val_set = torch.utils.data.random_split(train_dataset, [14000, 1000])

# Loading the test set
test_anomaly = CrackDataset(root='test_unbalanced/anomaly', transform=transform, label=0)
test_normal = CrackDataset(root='test_unbalanced/normal', transform=transform, label=1)
test_set = ConcatDataset([test_anomaly, test_normal])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)


def train_autoencoder(latent_dim):
    log_directory = f"experiments_dim_{latent_dim}"

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = Trainer(
        default_root_dir=log_directory,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=20,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, filename=f'checkpoint_{latent_dim}'),
        ],
    )

    if not os.path.isfile('checkpoint.ckpt'):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint('checkpoint.ckpt')
    else:
        model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result


def cluster(model):
    encoder_model = model.encoder
    encoder_model.eval()

    gm = GaussianMixture(n_components=2)
    x_train = []

    # Fit with the training data
    with torch.no_grad():
        for data in train_loader:
            inputs, label = data
            outputs = encoder_model.forward(inputs)
            x_train.extend(outputs.numpy())

    gm.fit(x_train)

    x_test = []
    y = []

    # Get predictions from test data
    with torch.no_grad():
        for data in test_loader:
            inputs, label = data
            outputs = encoder_model.forward(inputs)
            x_test.extend(outputs.numpy().astype(float))
            y.extend(label.numpy())

    predictions = gm.predict(x_test)
    y = np.array(y, dtype=np.int32)

    # Assign correct labels to clusters
    cluster_labels = {
        0: np.bincount(y[predictions == 0]).argmax(),
        1: np.bincount(y[predictions == 1]).argmax(),
    }
    predictions = [cluster_labels[prediction] for prediction in predictions]
    visualize_confusion_matrix(y, predictions, cluster_labels)
    print(precision_recall_fscore_support(y, predictions, average='micro'))


LATENT_DIM = 128

model, result = train_autoencoder(LATENT_DIM)
print(result)
visualize_reconstructions(model, get_images(test_set, 8))
cluster(model)
