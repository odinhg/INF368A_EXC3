import torch
from tqdm import tqdm
import numpy as np
from utilities import EarlyStopper
from os.path import join
from configfile import *

def train_classifier(classifier, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device):
    train_history = {"train_loss":[], "train_accuracy":[], "val_loss":[], "val_accuracy":[]}
    steps = len(train_dataloader) // 5 #Compute validation and train loss 5 times every epoch
    earlystop = EarlyStopper()
    for epoch in range(epochs):
        train_losses = []
        train_accuracies = []
        for i, data in enumerate((pbar := tqdm(train_dataloader))):
            images, labels  = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_accuracies.append((predicted == labels).sum().item() / labels.size(0))
            if i % steps == steps - 1:
                train_loss = np.mean(train_losses)
                train_losses = []
                train_accuracy = 100.0 * np.mean(train_accuracies)
                train_accuracies = []
                correct = 0
                total = 0
                val_losses = []
                classifier.eval()
                with torch.no_grad():
                    for data in val_dataloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = classifier(images)
                        val_losses.append(loss_function(outputs, labels).item())
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    val_accuracy = 100.0 * correct / total
                    val_loss = np.mean(val_losses)
                if train_history["val_accuracy"] and val_accuracy > np.max(train_history["val_accuracy"]):
                    torch.save(classifier.state_dict(), join(checkpoints_path, "best.pth"))
                train_history["train_loss"].append(train_loss)
                train_history["train_accuracy"].append(train_accuracy)
                train_history["val_loss"].append(val_loss)
                train_history["val_accuracy"].append(val_accuracy)
                pbar_string = f"Epoch {epoch}/{epochs-1} | Loss: Train={train_loss:.3f} Val={val_loss:.3f} | Acc.: Train={train_accuracy:.1f}% Val={val_accuracy:.1f}%"
                pbar.set_description(pbar_string)
                if earlystop(val_loss):
                    print(f"Early stopped at epoch {epoch}")
                    return train_history
                classifier.train()
    return train_history


def train_triplet(classifier, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device):
    train_history = {"train_loss":[], "val_loss":[]}
    steps = len(train_dataloader) // 5 #Compute validation and train loss 5 times every epoch
    negative_policy = "semi-hard"
    positive_policy = "easy"
    for epoch in range(epochs):
        train_losses = []
        val_loss = 0
        train_loss = 0

        if epoch >= 10: # Increase difficulty after some epochs to prevent collapse
            negative_policy = "hard"
        if epoch >= 25:
            positive_policy = "hard"
        
        for i, data in enumerate((pbar := tqdm(train_dataloader))):
            images, labels  = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = classifier(images, return_activations=True)

            loss = loss_function(outputs, labels, negative_policy, positive_policy)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            if i % steps == steps - 1:
                train_losses = []
                val_losses = []
                classifier.eval()
                with torch.no_grad():
                    for data in val_dataloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = classifier(images, return_activations=True)
                        val_losses.append(loss_function(outputs, labels, negative_policy="hard", positive_policy="hard").item())
                    val_loss = np.mean(val_losses)
                loss_function.mine_hard_triplets = False
                train_history["train_loss"].append(train_loss)
                train_history["val_loss"].append(val_loss)
                classifier.train()
            if train_losses:
                train_loss = np.mean(train_losses)
            pbar_string = f"Epoch {epoch}/{epochs-1} | TripletLoss: Train={train_loss:.3f} Val={val_loss:.3f}"
            pbar.set_description(pbar_string)
        torch.save(classifier.state_dict(), join(checkpoints_path, "best.pth"))
    return train_history

def train_arcface(classifier, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device):
    train_history = {"train_loss":[], "train_accuracy":[], "val_loss":[], "val_accuracy":[]}
    steps = len(train_dataloader) // 5 #Compute validation and train loss 5 times every epoch
    earlystop = EarlyStopper()
    for epoch in range(epochs):
        train_losses = []
        train_accuracies = []
        for i, data in enumerate((pbar := tqdm(train_dataloader))):
            images, labels  = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            embeddings, outputs = classifier(images, return_activations_and_output=True)
            weights = classifier.get_normalized_weights()
            loss = loss_function(embeddings, weights, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_accuracies.append((predicted == labels).sum().item() / labels.size(0))
            if i % steps == steps - 1:
                train_loss = np.mean(train_losses)
                train_losses = []
                train_accuracy = 100.0 * np.mean(train_accuracies)
                train_accuracies = []
                correct = 0
                total = 0
                val_losses = []
                classifier.eval()
                with torch.no_grad():
                    for data in val_dataloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        embeddings, outputs = classifier(images, return_activations_and_output=True)
                        weights = classifier.get_normalized_weights()
                        val_losses.append(loss_function(embeddings, weights, labels).item())
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    val_accuracy = 100.0 * correct / total
                    val_loss = np.mean(val_losses)
                if train_history["val_accuracy"] and val_accuracy > np.max(train_history["val_accuracy"]):
                    torch.save(classifier.state_dict(), join(checkpoints_path, "best.pth"))
                train_history["train_loss"].append(train_loss)
                train_history["train_accuracy"].append(train_accuracy)
                train_history["val_loss"].append(val_loss)
                train_history["val_accuracy"].append(val_accuracy)
                pbar_string = f"Epoch {epoch}/{epochs-1} | Loss: Train={train_loss:.3f} Val={val_loss:.3f} | Acc.: Train={train_accuracy:.1f}% Val={val_accuracy:.1f}%"
                pbar.set_description(pbar_string)
                if earlystop(val_loss):
                    print(f"Early stopped at epoch {epoch}")
                    return train_history
                classifier.train()
    return train_history
