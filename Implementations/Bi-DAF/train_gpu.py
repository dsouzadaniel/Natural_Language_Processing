# Python Libraries
import os
import sys
import time
from pathlib import Path

# Project Imports
from utils import constant, helper
from data import loader
from model import architecture

# External Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path().resolve()))

import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Datasets
training_dataset = loader.SquadLoader(path_to_csv=os.path.join(constant.DATASET_FOLDER, constant.TRAIN_FILE))
validation_dataset = loader.SquadLoader(path_to_csv=os.path.join(constant.DATASET_FOLDER, constant.VALID_FILE))
#
# # DataLoaders
training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

# Model

BIDAF = architecture.BiDAF_GPU().to(device)

optimizer = optim.SGD(BIDAF.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()

epoch_training_loss_collect = []
epoch_validation_loss_collect = []

lowest_validation_loss = 100000
best_BIDAF = None

for epoch in range(constant.EPOCHS):
    time_before_epoch = time.time()
    # Training Run
    BIDAF, optimizer, epoch_training_loss = helper.model_train_GPU(BIDAF, optimizer, training_dataloader)
    # Validation Run
    epoch_validation_loss = helper.model_evaluate(BIDAF, validation_dataset)

    epoch_training_loss_collect.append(epoch_training_loss)
    epoch_validation_loss_collect.append(epoch_validation_loss)
    time_after_epoch = round((time.time() - time_before_epoch)/3600, 2)

    # Print Results
    print("EPOCH: {0}\t TRAIN_LOSS : {1}\t VALID_LOSS : {2} \t EPOCH_TIME : {3} hrs".format(epoch, epoch_training_loss, epoch_validation_loss,time_after_epoch))

    if epoch_validation_loss < lowest_validation_loss:
        lowest_validation_loss = epoch_validation_loss
        best_BIDAF = BIDAF
        torch.save(best_BIDAF.state_dict(), 'BIDAF.pth')
        print("\tLowest Validation Loss! -> Model Saved!")


stop_time = time.time()
time_taken = stop_time - start_time
print("\n\nTraining Complete!\t Total Time: {0}\n\n".format(time_taken))

print("Writing the Loss Graph")
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epoch_training_loss_collect, label='training')
plt.plot(epoch_validation_loss_collect, label='validation')
plt.legend(loc="upper right")
plt.savefig("loss.png")