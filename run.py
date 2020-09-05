from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sacred import SETTINGS

import torch
import torch.nn as nn
from torchvision import transforms as T, models
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import os
import numpy as np
from dataset import CUB
from PIL import Image

MONGODB_USER = 'example'
MONGODB_PASS = 'example'
MONGODB_URL = 'example.com:27017'
SETTINGS.CAPTURE_MODE = 'sys'
ex = Experiment(name="sacred-example")
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(MongoObserver(
    url=f"mongodb://{MONGODB_USER}:{MONGODB_PASS}@{MONGODB_URL}",
    db_name="sacred"
))

@ex.config
def configuration():
    lr = 1e-2
    max_epoch = 200
    batch_size = 32
    dataroot = os.path.join(os.environ["DATAROOT"], "CUB_200_2011")
    device_id = "cuda:0"

def criterion(output, target):
    return nn.functional.cross_entropy(output, target)

def evaluation(model, loader, device):
    model.eval()
    running_loss = 0.
    running_tp = 0
    running_total = 0
    processbar = tqdm(total=len(loader), leave=False)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = criterion(output, target)
        running_loss += loss.item()

        pred = output.argmax(dim=1)
        tp = (pred == target).sum().item()
        total = pred.shape[0]
        running_tp += tp
        running_total += total

        processbar.update(1)

    processbar.close()
    mean_loss  = running_loss / len(loader)
    mean_acc = 1. * running_tp / running_total

    return (mean_loss, mean_acc)


@ex.automain
def main(lr, max_epoch, batch_size, dataroot, device_id, _run):
    device = torch.device(device_id)

    train_transforms = T.Compose([
        T.Resize((256, 256), interpolation=Image.NEAREST),
        T.RandomResizedCrop((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transforms = T.Compose([
        T.Resize((224, 224), interpolation=Image.NEAREST),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_dataset = CUB(dataroot, train=True, transform=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    test_dataset = CUB(dataroot, train=False, transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)
    train_test_dataset = CUB(dataroot, train=True, transform=test_transforms)
    train_test_dataloader = DataLoader(train_test_dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=2048, out_features=200, bias=True)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    for epoch_idx in range(max_epoch):
        model.train()

        running_loss = 0.
        processbar = tqdm(total=len(train_dataloader), leave=False)
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            running_loss += loss.item()
            processbar.set_postfix({"Loss r.": loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            processbar.update(1)
        processbar.close()
        mean_loss = running_loss / len(train_dataloader)
        _run.log_scalar("training.loss", mean_loss, epoch_idx)

        eval_trainset_loss, eval_trainset_acc = evaluation(model, train_test_dataloader, device)
        eval_testset_loss, eval_testset_acc = evaluation(model, test_dataloader, device)

        _run.log_scalar("eval.trainset.loss", eval_trainset_loss, epoch_idx)
        _run.log_scalar("eval.testset.loss", eval_testset_loss, epoch_idx)
        _run.log_scalar("eval.trainset.acc", eval_trainset_acc, epoch_idx)
        _run.log_scalar("eval.testset.acc", eval_testset_acc, epoch_idx)

        print(f"Finish {epoch_idx}!")
