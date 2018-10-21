# Data Prep Script for Preparing Penn Treebank Dataset

# Libraries
import os
from random import shuffle

# Paths
treebank_wsj_path = './DATA/treebank2/tagged/wsj/'
train_test_split = 0.8


# Helper Functions
def write_data_to_file(data_file, data_file_name):
    ''' This Function will write a file( newline format ) for every item in the list object '''
    with open(data_file_name, 'w+') as f:
        for line in data_file:
            f.write(line + '\n')
    return

def verify_train_test_splits(all_data_file, train_data_file, test_data_file):
    ''' A simple function to verify the correct split of test and train data '''
    if not (len(all_data_file)==(len(train_data_file)+len(test_data_file))) \
    or not (all_data_file[0]==train_data_file[0]) \
    or not (all_data_file[len(train_data_file)-1]==train_data_file[-1]) \
    or not (all_data_file[len(train_data_file)]==test_data_file[0]) \
    or not (all_data_file[-1]==test_data_file[-1]):
        return False
    else:
        return True

def parse_out_lines(temp_f):
    ''' This function will parse out a single Penn Treebank file and convert them into a single list of lines.'''
    all_lines = []
    this_line = []
    for single in temp_f:
        if len(single) == 0 or "===" in single:
            continue
        single = single.strip('[ ]')
        #     print('*',single,"\n\n\n")
        for sub_term in single.split(' '):
            if '/' in sub_term:
                word, tag = ''.join(sub_term.split('/')[:-1]), sub_term.split('/')[-1]
                if word == '.':
                    this_line.append(word + '|' + tag)
                    all_lines.append(' '.join(this_line))
                    this_line = []
                else:
                    this_line.append(word + '|' + tag)
    #     print("Number of lines is :",len(all_lines))
    return all_lines

print("Munging the Penn Treebank 2 Files...")
all_lines_total = []
# Iterate through the folder structure and apply function: parse_out_lines
for subdirs, dirs, files in os.walk(treebank_wsj_path):
    for indi_file in files:
        if indi_file.endswith('.pos'):
            with open(os.path.join(subdirs, indi_file)) as f:
                all_lines_total.extend(parse_out_lines(f.read().splitlines()))

# Shuffle the Lines
shuffle(all_lines_total)

# Calculate the index to split on
split_index = round(len(all_lines_total) * train_test_split)
# Train & Test Splits
train_lines = all_lines_total[:split_index]
test_lines = all_lines_total[split_index:]


# Verify Train and Test files
if verify_train_test_splits(all_lines_total, train_lines, test_lines):
    # Write Train and Test files
    write_data_to_file(train_lines, './DATA/train_lines.txt')
    write_data_to_file(test_lines, './DATA/test_lines.txt')

    print("Train and Test sets Created!")

