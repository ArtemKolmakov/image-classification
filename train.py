import argparse
import torchvision

from dataset import create_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train loop for classification net.')
    parser.add_argument('num_classes', type=int, help='a number of classes in dataset')
    parser.add_argument('csv_file', type=str, help='path to csv file that contains dataset info in format:\n1.jpg,0\n2.jpg,1')
    parser.add_argument('root_dir', type=str, help='path to root dir with dataset')
    args = parser.parse_args()
    
    num_classes = args.num_classes
    csv_file = args.csv_file
    root_dir = args.root_dir

    train_loader, val_loader = create_dataloader(
        csv_file=csv_file,
        root_dir=root_dir,
        num_classes=num_classes
    )
    model = torchvision.models.resnext50_32x4d(pretrained=True)
    model= FineTuneModel(model, num_classes=num_classes)
    model = Model(model=model)
    model.train(train_loader, val_loader)
