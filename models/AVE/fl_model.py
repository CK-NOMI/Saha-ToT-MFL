import copy
import json
import logging
import os

import load_data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

                   
lr = 0.003
momentum = 0.9
log_interval = 10
loss_thres = 0.001

                                                             
FEATURE_DIM = int(os.environ.get("AVE_FEATURE_DIM", "96"))
NUM_CLASSES = int(os.environ.get("AVE_NUM_CLASSES", "28"))

               
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Generator(load_data.Generator):
    """Generator for AVE leaf-format dataset."""

    def read(self, path):
        global FEATURE_DIM, NUM_CLASSES

        self.trainset = {"users": [], "user_data": {}, "num_samples": []}
        self.valset = {"users": [], "user_data": {}, "num_samples": []}
        self.testset = {"users": [], "user_data": {}, "num_samples": []}

        train_file = os.path.join(path, "train.json")
        with open(train_file, "r", encoding="utf-8") as f:
            logging.info("loading %s", train_file)
            data = json.load(f)
            self.trainset["users"] += data["users"]
            self.trainset["user_data"].update(data["user_data"])
            self.trainset["num_samples"] += data["num_samples"]

        val_file = os.path.join(path, "val.json")
        if not os.path.isfile(val_file):
            raise FileNotFoundError(
                "AVE validation partition is missing: {}. Re-run "
                "scripts/prepare_ave_partitions.py; final test data must not be used as validation."
                .format(val_file)
            )
        with open(val_file, "r", encoding="utf-8") as f:
            logging.info("loading %s", val_file)
            data = json.load(f)
            self.valset["users"] += data["users"]
            self.valset["user_data"].update(data["user_data"])
            self.valset["num_samples"] += data["num_samples"]

        test_file = os.path.join(path, "test.json")
        with open(test_file, "r", encoding="utf-8") as f:
            logging.info("loading %s", test_file)
            data = json.load(f)
            self.testset["users"] += data["users"]
            self.testset["user_data"].update(data["user_data"])
            self.testset["num_samples"] += data["num_samples"]

        labels = []
        trainset_size = 0
        inferred_dim = None
        for user in self.trainset["users"]:
            udata = self.trainset["user_data"][user]
            ys = list(udata.get("y", []))
            xs = list(udata.get("x", []))
            labels += ys
            trainset_size += len(ys)
            if inferred_dim is None and len(xs) > 0:
                inferred_dim = len(xs[0])

        self.labels = sorted(list(set(labels)))
        self.trainset_size = trainset_size

        if inferred_dim is not None and inferred_dim > 0:
            FEATURE_DIM = int(inferred_dim)
        if len(self.labels) > 0:
            NUM_CLASSES = int(max(self.labels)) + 1

        logging.info(
            "AVE generator ready: users=%s train_samples=%s feature_dim=%s num_classes=%s",
            len(self.trainset["users"]), self.trainset_size, FEATURE_DIM, NUM_CLASSES
        )

    def generate(self, path):
        self.read(path)
        return self.trainset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(FEATURE_DIM, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def _to_dataset(data_dict):
    x = np.array(data_dict["x"], dtype=np.float32)
    y = np.array(data_dict["y"], dtype=np.int64)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    return TensorDataset(x_tensor, y_tensor)


def get_trainloader(trainset, batch_size):
    dataset = _to_dataset(trainset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_evalloader(evalset, batch_size):
    dataset = _to_dataset(evalset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_testloader(testset, batch_size):
    """Backward-compatible alias for a non-training evaluation loader."""
    return get_evalloader(testset, batch_size)


def extract_weights(model):
    weights = []
    state_dict = model.to(torch.device("cpu")).state_dict()
    for name in state_dict.keys():
        weight = state_dict[name]
        weights.append((name, weight))
    return weights


def load_weights(model, weights):
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight
    model.load_state_dict(updated_state_dict, strict=False)


def flatten_weights(weights):
    weight_vecs = []
    for _, weight in weights:
        weight_vecs.extend(weight.flatten().tolist())
    return np.array(weight_vecs)


def extract_grads(model):
    grads = []
    for name, weight in model.to(torch.device("cpu")).named_parameters():
        if weight.requires_grad:
            grads.append((name, weight.grad))
    return grads


def train(model, trainloader, optimizer, epochs, reg=None, rho=None):
    old_model = copy.deepcopy(model)
    old_model.to(device)
    old_model.eval()

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_gw_l2_loss = 0.0, 0.0
        correct = 0
        for batch_id, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            if reg is not None and rho is not None:
                gw_l2_loss = 0.0
                for param_a, param_b in zip(model.parameters(), old_model.parameters()):
                    gw_l2_loss += rho / 2 * torch.sum(torch.square(param_a - param_b.detach()))
                loss += gw_l2_loss
                train_gw_l2_loss += gw_l2_loss.item()

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_id % log_interval == 0:
                logging.debug("Epoch: [%s/%s]\tLoss: %.6f", epoch, epochs, loss.item())

            if loss.item() < loss_thres:
                return loss.item()

            _, predicted = logits.max(1)
            correct += predicted.eq(y).sum().item()

    total = len(trainloader.dataset)
    train_loss = train_loss / len(trainloader)
    accuracy = correct / max(1, total)
    logging.debug("Train accuracy: %s", accuracy)

    if reg is not None and rho is not None:
        train_gw_l2_loss = train_gw_l2_loss / len(trainloader)
        logging.info("loss: %s l2_loss: %s", train_loss, train_gw_l2_loss)
    else:
        logging.info("loss: %s", train_loss)

    return train_loss


def evaluate(model, evalloader):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)

    eval_loss = 0.0
    correct = 0
    total = len(evalloader.dataset)
    with torch.no_grad():
        for x, y in evalloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            eval_loss += criterion(logits, y).item()
            _, predicted = logits.max(1)
            correct += predicted.eq(y).sum().item()

    eval_loss = eval_loss / len(evalloader)
    accuracy = correct / max(1, total)
    logging.debug("Evaluation loss: %s Accuracy: %.2f%%", eval_loss, 100 * accuracy)
    return eval_loss, accuracy


def test(model, testloader):
    """Backward-compatible alias. Call evaluate() for validation or final test explicitly."""
    return evaluate(model, testloader)
