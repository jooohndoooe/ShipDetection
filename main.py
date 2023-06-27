import matplotlib.pyplot as plt
from train import train
from common import *
from data import get_submission

# If to_train is set to True trains and evaluates the model
to_train = False
# If to_load is set to True loads a trained model from the 'models' folder
to_load = True

if to_load:
    net = torch.load('models/UNet_trained.pt', map_location=torch.device('cpu'))
    net.to(device)

if to_train:
    train_acc, test_acc, train_loss, test_loss = train()

    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(test_acc, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Testing Accuracies')
    plt.show()

    plt.plot(test_loss, label='Testing Loss')
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Testing Losses')
    plt.show()

# Generating a submission using the model and 'test_v2' data
submission = get_submission()
submission.to_csv('data/submission.csv', index=False)
