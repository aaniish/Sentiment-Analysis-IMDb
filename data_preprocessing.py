import numpy as np
import zipfile
import pandas as pd

# Load and preprocess the data
with zipfile.ZipFile("data.zip", "r") as zip_ref:
    with zip_ref.open("data/FT.data") as ft_file, zip_ref.open("data/LABEL.data") as label_file:
        # Since columns vary, using lists to store data
        X = [list(map(int, line.strip().decode().split(','))) for line in ft_file]
        y = [int(line.strip()) for line in label_file]

# Create a list of dictionaries to store the word frequencies (NOTE: decided to process data similar to lab07, where in each row
# I record the frequency of the word indices. For example in column 2 of row 1 in train.data I recorded 21 because 2 appeared
# 21 times in the first row of FT.data)
word_frequencies = []

for row in X:
    word_count = {}
    for word in row:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    word_frequencies.append(word_count)

# Create a dataframe from the word frequencies
df = pd.DataFrame(word_frequencies).fillna(0)

# Sort the columns based on the column names (keys) in ascending order
df = df.sort_index(axis=1)

# Attach the labels to the instances
df['label'] = y

# Split the data into training and testing sets
train_data = df.iloc[:20000]
test_data = df.iloc[20000:]

# Save the training and testing data (NOTE: Test data was compressed into zip file because the file was too large. Parts 1 and 2
# deal with unzipping automatically)
train_data.to_csv("train.data", sep=" ", index=False, header=False)
test_data.to_csv("test.data", sep=" ", index=False, header=False)
