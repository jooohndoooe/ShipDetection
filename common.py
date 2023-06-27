import math
import torch
from UNet import UNet

# If local is set to True sets data paths to work locally, if set to False sets them to work on Kaggle
local = True

if local:
    TRAIN_DATA = 'data/train_v2/'
    TEST_DATA = 'data/test_v2/'
    DATASET = 'data/train_ship_segmentations_v2.csv'
else:
    TRAIN_DATA = '/kaggle/input/airbus-ship-detection/train_v2/'
    TEST_DATA = '/kaggle/input/airbus-ship-detection/test_v2/'
    DATASET = '/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv'


IMAGE_HEIGHT = 768
IMAGE_WIDTH = 768
TEST_SIZE = 0.02
BATCH_SIZE = 4
NUM_BATCHES_PER_EPOCH = 250
LEARNING_RATE = .001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 10


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = UNet()


# Default progress bar
def progress_bar(progress, total):
    percent = math.ceil(100 * (progress / float(total)))
    bar = '■' * int(percent) + '-' * (100 - int(percent))
    print(f"\r|{bar}| {percent:.2f}%", end="")


# Progress bar with loss printed at the end
def training_progress_bar(progress, total, loss):
    percent = math.ceil(100 * (progress / float(total)))
    bar = '■' * int(percent) + '-' * (100 - int(percent))
    print(f"\r|{bar}| {percent:.2f}% loss:{loss}", end="")


# Calculates dice score of two tensors
def dice_score(y, yHat):
    threshold = 0.5
    y_mask = (y > threshold).float()
    yHat_mask = (yHat > threshold).float()
    intersection = torch.sum(y_mask * yHat_mask)
    union = torch.sum(y_mask) + torch.sum(yHat_mask)
    score = (2.0 * intersection) / (union + 1e-8)
    return score.cpu().numpy()
