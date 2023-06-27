# ShipDetection
 Solution for Airbus Ship Detection Challenge

The solution consists of 2 folders and 5 Python files.

Folders:

- 'docs' folder contains three files:
 1. 'task.docs' - task description;
 2. 'requirements.txt' - file with required python modules;
 3. 'data_analysis.ipynb' - jupyter notebook with exploratory data analysis of the dataset;

- 'models' folder contains trained networks

Files:

- 'main.py' runs the program
- 'common.py' stores common variables, constants, and functions
- 'data.py' contains data generators and methods used to work with data
- 'UNet.py' contains the neural network class
- 'train.py' contains the code used to train the network

# Step-by-step solution:

First, we read the 'train_ship_segmentation_v2.csv', drop rows with null data in order to have only images with ships on them, and create a dictionary with 'ImageId' as the key and concatenation of all 'EncodedPixels' strings with that id as the value.

Next, we split the dataframe we got from the csv file into a train and test parts.

Then we create two data generators - for train and test accordingly.

Next, we train a network, created based on UNet architecture.

Training is done in batches of 4, consists of 10 epcohs, 250 batches each. We use BCEWithLogits as a loss function and the ADAM optimizer. To evaluate the network we use dice score.

After training is completed we plot the loss and accuracy of the model on train and test data. In the end, we get a loss of approximately 0.015 and a dice score of 0.275

Next, we use the trained network on the images provided for the submission, encode the result, and save to csv.

After uploading the result to Kaggle we get a score of  0.63994
