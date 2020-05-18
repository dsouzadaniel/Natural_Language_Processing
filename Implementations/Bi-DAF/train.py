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


# Datasets
training_dataset = loader.SquadLoader(path_to_csv=os.path.join(constant.DATASET_FOLDER, constant.TRAIN_FILE))
validation_dataset = loader.SquadLoader(path_to_csv=os.path.join(constant.DATASET_FOLDER, constant.VALID_FILE))
#
# # DataLoaders
# training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=False)
# validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

# Model

BIDAF = architecture.BiDAF()

optimizer = optim.SGD(BIDAF.parameters(), lr=0.005, momentum=0.9)

start_time = time.time()

epoch_training_loss_collect = []
epoch_validation_loss_collect = []

lowest_validation_loss = 100000
best_BIDAF = None

for epoch in range(constant.EPOCHS):
    # Training Run
    BIDAF, optimizer, epoch_training_loss = helper.model_train(BIDAF, optimizer, training_dataset)
    # Validation Run
    epoch_validation_loss = helper.model_evaluate(BIDAF, validation_dataset)

    epoch_training_loss_collect.append(epoch_training_loss)
    epoch_validation_loss_collect.append(epoch_validation_loss)

    # Print Results
    print("EPOCH: {0}\t TRAIN_LOSS : {1}\t VALID_LOSS : {2} ".format(epoch, epoch_training_loss, epoch_validation_loss))

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