import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FineTuneModel(nn.Module):
    def __init__(self, original_model, num_classes):
        super(FineTuneModel, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Linear(2048, num_classes)
        self.modelName = 'Resnet50'
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        f = self.features(x)        
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y
    
class ModelInterface:
    def __init__(self, model, device='cuda:0'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        self.criterion= nn.BCEWithLogitsLoss(reduction='mean')
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=0.3,
            patience=1,
            verbose=True,
            mode='max',
        )
    
    def _step(self, data_loader, is_train=False):
        losses = []
        for batch in data_loader:
            img, labels = batch.values()
            img, labels= (
                img.to(self.device),
                labels.to(self.device),
            )
            model_out = self.model.forward(img)
            loss = self.criterion(model_out, labels.type_as(model_out))
            losses.append(loss.item())
            if is_train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        return np.mean(losses)
    
    def train(self, train_dataloader, val_dataloader, save_folder='./checkpoints/', num_epochs=240):
        best_loss = 100
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = self._step(
                train_dataloader,
                is_train=True,
            )
            with torch.no_grad():
                val_loss = self._step(
                    val_dataloader,
                )
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model, 'classifer.pth')
            print(f'epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}')
            self.scheduler.step(val_loss)