import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from os.path import isfile
from tqdm import tqdm
from configfile import *
from dataloader import FlowCamDataLoader
from backbone import BackBone

if __name__ == "__main__":
    #Use custom backbone based on EfficientNet v2
    #number_of_classes = len(class_names)
    #classifier = BackBone(number_of_classes)
    classifier.to(device)

    if not isfile(join(checkpoints_path, "best.pth")):
        print("No checkpoint found! Please run training before evaluating model.")
    else:
        print("Loading checkpoint.")
        classifier.load_state_dict(torch.load(join(checkpoints_path, "best.pth")))
        classifier.eval()

        correct_pred = {classname: 0 for classname in class_names}
        total_pred = {classname: 0 for classname in class_names}
        
        with torch.no_grad():
            print("Evaluating model on test data.")
            for data in tqdm(test_dataloader):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = classifier(images)
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[class_names[label]] += 1
                    total_pred[class_names[label]] += 1
        
        #Calculate accuracies
        correct = 0
        total = 0
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * correct_count / total_pred[classname]
            print(f"\"{classname}\" {accuracy:.2f}%")
            total += total_pred[classname]
            correct += correct_count
        total_accuracy = 100 * correct / total
        print(f"Total accuracy: {total_accuracy:.2f}%")
