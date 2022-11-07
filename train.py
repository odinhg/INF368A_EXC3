import torch
import numpy as np
import pandas as pd
from os.path import isfile, join
from tqdm import tqdm
from configfile import *
from utilities import save_train_plot, save_loss_plot
from dataloader import FlowCamDataLoader
from trainer import train_classifier, train_triplet, train_arcface
from torchsummary import summary

if __name__ == "__main__":
    #Use custom backbone based on EfficientNet v2
    summary(classifier, (3, *image_size), device=device)
    classifier.to(device)

    print("Training...")
    if model_type == "triplet":
        train_history = train_triplet(classifier, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)
        save_loss_plot(join(figs_path, "training_plot.png"), train_history)
    elif model_type == "arcface":
        train_history = train_arcface(classifier, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)
        save_train_plot(join(figs_path, "training_plot.png"), train_history)
    else:
        train_history = train_classifier(classifier, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)
        save_train_plot(join(figs_path, "training_plot.png"), train_history)
