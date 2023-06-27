import numpy as np
import pandas as pd
import cv2
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from common import *


train_ship_segmentation = pd.read_csv(DATASET)

# Drop images with no ships for training
train_ship_segmentation = train_ship_segmentation.dropna()

# Split the training dataset into train and test
train, test = train_test_split(train_ship_segmentation, test_size=TEST_SIZE)


# Creating a dictionary with 'ImageId' as key and a concatenation of all 'EncodedPixels' string with that id
print('Processing training data:')
joined_strings = defaultdict(str)
i = 0
for _, data in train_ship_segmentation.iterrows():
    i += 1
    progress_bar(i, len(train_ship_segmentation))
    id_ = data["ImageId"]
    text = data["EncodedPixels"]
    joined_strings[id_] += text + " "
progress_bar(1, 1)


# Decoded the 'EncodedPixels' string to form a 2d-array mask
def decode(rle_list, shape):
    tmp_flat = np.zeros(shape[0] * shape[1])
    if len(rle_list) == 1:
        mask = np.reshape(tmp_flat, shape).T
    else:
        start = rle_list[::2]
        length = rle_list[1::2]
        for i, v in zip(start, length):
            tmp_flat[(int(i) - 1):(int(i) - 1) + int(v)] = 255
        mask = np.reshape(tmp_flat, shape).T
    return mask


# Encodes a 2d tensor using rle encoding
def encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# Data generator, that generates a batch tensors of given size from a given dataset
def data_generator(dataset, image_width, image_height, batch_size):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    while True:
        for start_idx in range(0, len(dataset), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            data = []
            target = []

            for idx in batch_indices:
                row = dataset.iloc[idx]
                image_id = row['ImageId']
                path = TRAIN_DATA + image_id
                img = cv2.imread(path)
                img = cv2.resize(img, (image_width, image_height))
                data.append(np.array(img))
                target.append(cv2.resize(decode(joined_strings[image_id].split(), (IMAGE_WIDTH, IMAGE_HEIGHT)),
                                         (image_width, image_height)))

            data = np.array(data) / 255
            target = np.array(target) / 255
            data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
            target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

            yield data, target


# Creating two data generators: for train and test respectively
train_generator = data_generator(train, IMAGE_WIDTH, IMAGE_HEIGHT, batch_size=BATCH_SIZE)
test_generator = data_generator(test, IMAGE_WIDTH, IMAGE_HEIGHT, batch_size=BATCH_SIZE)


# Generates a submission csv
def get_submission():
    submission_ids = os.listdir(TEST_DATA)
    submission = pd.DataFrame({'ImageId': submission_ids, 'EncodedPixels': None})

    print('\nProcessing submission images')
    for index, row in submission.iterrows():
        progress_bar(index, len(submission))
        image_id = row['ImageId']
        path = TEST_DATA + image_id
        img = cv2.imread(path)
        data = np.array([img]) / 255
        data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
        y = net(data)[0][0]
        threshold = 0.5
        y_mask = (y > threshold).float()
        submission.at[index, 'EncodedPixels'] = encode(y_mask)
    progress_bar(1, 1)
    return submission

