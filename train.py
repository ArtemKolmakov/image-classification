import argparse
import torchvision

from model_wrapped import FineTuneModel, ModelInterface
from dataset import create_dataloader


def get_params():
    parser = argparse.ArgumentParser(description='train loop for classification net.')
    parser.add_argument('csv_file', type=str, help='path to csv file that contains dataset info in format:\n1.jpg,0\n2.jpg,1')
    parser.add_argument('root_dir', type=str, help='path to root dir with dataset')  
    parser.add_argument('path_to_model', type=str, default='', help='path to pretraned model')
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_params()
    
    num_classes = args.num_classes
    csv_file = args.csv_file
    root_dir = args.root_dir
    path_to_model = args.path_to_model

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
    model.train(train_loader, val_loader)
