# External Libraries
import pandas as pd
import ast
from torch.utils.data import Dataset, DataLoader
import os
DATASET_FOLDER = 'dataset'
TRAIN_FILE = 'train.csv'
VALID_FILE = 'valid.csv'

class SquadLoader(Dataset):
    def __init__(self, path_to_csv):
        self.data_tuples = []
        df = pd.read_csv(path_to_csv)
        self.data_tuples = [(r['CONTEXT'], r['QUERY'], r['POSITIONS']) for _, r in df.iterrows()]
        self.data_tuples = [(r[0], r[1], ast.literal_eval(r[2])) for r in self.data_tuples]

    def __getitem__(self, index):
        return self.data_tuples[index]

    def __len__(self):
        return len(self.data_tuples)

# # Datasets
# training_dataset = SquadLoader(path_to_csv=os.path.join(DATASET_FOLDER, TRAIN_FILE))
# validation_dataset = SquadLoader(path_to_csv=os.path.join(DATASET_FOLDER, VALID_FILE))
#
# # DataLoaders
# training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=False)
# validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
#
#
# for t in training_dataloader:
#     print(t)
#     a,b,c = t
#     break