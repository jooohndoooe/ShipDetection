import torch.nn as nn
import numpy as np
from common import *
from data import train_generator, test_generator

# If set to true saves the model to the 'models' folder after finishing training
save = True


def train():
    print("\nTraining the model")
    net.to(device)

    # We use BCEWithLogits as a loss function and the ADAM optimizer
    loss_fun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Loss and accuracy of train and test data
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []

    # The loss of the final use of the loss function, used for the progress bar display
    final_loss = -1

    # Iterating through epochs
    for epoch_i in range(NUM_EPOCHS):
        # Set the network to training mode
        net.train()

        # Loss and accuracy of the batch
        batch_acc = []
        batch_loss = []

        # Iterating through the batch
        for batch_idx, (X, y) in enumerate(train_generator):

            # Forward pass
            X = X.to(device)
            y = y.to(device)
            yHat = net(X)
            loss = loss_fun(yHat, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_acc.append(dice_score(y, yHat))
            batch_loss.append(loss.item())
            final_loss = loss.item()
            training_progress_bar(batch_idx + epoch_i * NUM_BATCHES_PER_EPOCH * BATCH_SIZE, NUM_EPOCHS *
                                  NUM_BATCHES_PER_EPOCH * BATCH_SIZE, loss.item())

            # Exit the generator after desired number of batches
            if batch_idx == NUM_BATCHES_PER_EPOCH:
                break

        train_loss.append(np.mean(batch_loss))
        train_acc.append(np.mean(batch_acc))

        # Set the network to evaluation mode and evaluate using test generator
        net.eval()
        X, y = next(test_generator)
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)
            yHat = net(X)
            test_acc.append(dice_score(y, yHat))
            loss = loss_fun(yHat, y)
            test_loss.append(loss.item())

    training_progress_bar(1, 1, final_loss)

    if save:
        model = torch.jit.script(net)
        model.save('models/UNet.pt')
    return train_acc, test_acc, train_loss, test_loss
