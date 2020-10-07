import warnings
warnings.simplefilter("ignore")

import torchvision
from fastai.vision.all import *
from fastbook import *

from model_wrapped import FineTuneModel, ModelInterface
from dataset import create_dataloader


if __name__ == "__main__":
    path = untar_data(URLs.MNIST_SAMPLE)
    Path.BASE_PATH = path
    csv_file = path.ls()[1]
    root_dir = path
    path_to_model = ''
    num_epoch=10

    train_loader, val_loader, num_classes = create_dataloader(
        csv_file=csv_file,
        root_dir=root_dir
    )
    if path_to_model:
        model = torch.load(path_to_model)
    else:
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        model= FineTuneModel(model, num_classes=num_classes)
    model = ModelInterface(model=model)
    model.train(train_loader, val_loader, num_epochs=num_epoch)
