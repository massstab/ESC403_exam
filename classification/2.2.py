import logging
import os
import numpy as np
import sys
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Silence tensorflow a bit
np.set_printoptions(threshold=sys.maxsize)

from contextlib import redirect_stdout
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from preparation import X, y, X_pred, img_name
import scipy.io
from helpers2 import create_model


# TODO: ChestCT and AbdomentCT maybe are not normalized correctly?

save_to_file = False
preview = False
run_rf = True
run_cnn = False

# Global variables and classes definition
classes = {'AbdomenCT': 0, 'BreastMRI': 1, 'ChestCT': 2, 'CXR': 3, 'Hand': 4, 'HeadCT': 5}
BATCH_SIZE = 64
N_CLASSES = 6
EPOCHS = 10

class_names = list(classes.keys())

# Creating train, test and validation sets after shuffling the set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=False)

print(X_test.shape, y_test.shape)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)


# Take a look at some images
if preview:
    plt.figure(figsize=(5, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_train[i], cmap=plt.cm.hot)
        plt.xlabel(class_names[y_train[i]])
        plt.xticks([])
        plt.yticks([])
    plt.show()

if run_rf:
    X_train_rf = X_train.reshape((X_train.shape[0], X_train.shape[1] * X.shape[2] * X.shape[3]))
    X_test_rf = X_test.reshape((X_test.shape[0], X_test.shape[1] * X.shape[2] * X.shape[3]))
    X_pred_rf = X_pred.reshape((X_pred.shape[0], X_pred.shape[1] * X.shape[2] * X.shape[3]))

    model_rf = RandomForestClassifier(bootstrap=True)
    model_rf.fit(X_train_rf, y_train)

    pred_test = model_rf.predict(X_test_rf)
    print("Accuracy random forest:", accuracy_score(y_test, pred_test))

    predictions = model_rf.predict(X_pred_rf)

    df = pd.DataFrame(list(zip(img_name, [class_names[pred] for pred in predictions], predictions)), columns=['img', 'class name', 'category'])
    if save_to_file:
        df.to_csv('../report/data/predictions_rf.csv')

if run_cnn:
    # model_conv = create_model()
    # model_conv.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

    # model_conv.save(f'saved_model/conv_ep{EPOCHS}_batch{BATCH_SIZE}')
    model_conv = keras.models.load_model('saved_model/conv')
    loss_conv, accuracy_conv = model_conv.evaluate(X_test, y_test)
    with open('saved_model/conv_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model_conv.summary()

    # Saves a cool looking plot of the model
    # keras.utils.plot_model(model_conv, "model_mnist_conv.png", show_shapes=True)

    print("Test Accuracy CNN; ", accuracy_conv)
    predictions = model_conv.predict(X_pred)
    predictions = np.argmax(predictions, axis=1)
    df = pd.DataFrame(list(zip(img_name, [class_names[pred] for pred in predictions], predictions)), columns=['img', 'class name', 'category'])
    if save_to_file:
        df.to_csv('../report/data/predictions_cnn.csv')

    plt.figure(figsize=(10, 10))
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.imshow(X_pred[i], cmap=plt.cm.hot)
        plt.xlabel(class_names[df['category'].iloc[i]])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    if save_to_file:
        plt.savefig('../report/images/pred_img_cnn.png')
    # plt.show()
