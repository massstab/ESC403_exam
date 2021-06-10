import numpy as np
import os
from tqdm import tqdm
import cv2

# Data paths
# Here we define the paths of the classes folder after unzipping the dataset
data_dir = "data2/med-mnist"
abdomen_DIR = data_dir + '/AbdomenCT'
breastmri_DIR = data_dir + '/BreastMRI'
chestct_DIR = data_dir + '/ChestCT'
cxr_DIR = data_dir + '/CXR'
hand_DIR = data_dir + '/Hand'
headct = data_dir + '/HeadCT'

# Global variables and classes definition
classes = {'AbdomenCT': 0, 'BreastMRI': 1, 'ChestCT': 2, 'CXR': 3, 'Hand': 4, 'HeadCT': 5}
BATCH_SIZE = 64
N_CLASSES = 6
EPOCHS = 5


# Append of the data

# Here we append the data from all classes by using the function: "make_train_data".
# The labels are collected for each class using the function "assign_label"
def assign_label(img, class_type):
    return class_type


X = []
y = []


def make_train_data(class_type, DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img, class_type)
        path = os.path.join(DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        X.append(np.array(img))
        y.append(label)


make_train_data(classes.get('AbdomenCT'), abdomen_DIR)
make_train_data(classes.get('BreastMRI'), breastmri_DIR)
make_train_data(classes.get('ChestCT'), chestct_DIR)
make_train_data(classes.get('CXR'), cxr_DIR)
make_train_data(classes.get('Hand'), hand_DIR)
make_train_data(classes.get('HeadCT'), headct)

y = np.array(y)
X = np.array(X)

X_pred = []
img_name = []
for img in tqdm(os.listdir('data2/unlabeled_Med_MNIST')):
    path = os.path.join('data2/unlabeled_Med_MNIST', img)
    img_name.append(img)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    X_pred.append(np.array(img))

X_pred = np.array(X_pred)
