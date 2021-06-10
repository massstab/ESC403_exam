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
run_rf = True
run_cnn = False

# Global variables and classes definition
classes = {'AbdomenCT': 0, 'BreastMRI': 1, 'ChestCT': 2, 'CXR': 3, 'Hand': 4, 'HeadCT': 5}
BATCH_SIZE = 64
N_CLASSES = 6
EPOCHS = 5


# Creating train, test and validation sets after shuffling the set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=False)

print(X_test.shape, y_test.shape)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

# Take a look at some images
plt.figure(figsize=(10, 10))
class_names = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_train[i], cmap=plt.cm.hot)
    plt.xlabel(class_names[y_train[i]])
    plt.xticks([])
    plt.yticks([])
plt.show()

if run_rf:
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X.shape[2] * X.shape[3]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X.shape[2] * X.shape[3]))

    model_rf = RandomForestClassifier(bootstrap=True)
    model_rf.fit(X_train, y_train)

    predictions = model_rf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))

    df = pd.DataFrame(list(zip(img_name, [class_names[pred] for pred in predictions], predictions)), columns=['img', 'class name', 'category'])
    print(df)
    if save_to_file:
        df.to_csv('../report/data/rf_predictions.csv')

if run_cnn:
    # model_conv = create_model()
    # model_conv.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))
    #
    # model_conv.save('saved_model/conv')
    model_conv = keras.models.load_model('saved_model/conv')
    loss_conv, accuracy_conv = model_conv.evaluate(X_test, y_test)
    with open('saved_model/conv_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model_conv.summary()

    # Saves a cool looking plot of the model
    # keras.utils.plot_model(model_conv, "model_mnist_conv.png", show_shapes=True)

    print("Test Accuracy_cov =", accuracy_conv)
    predictions = model_conv.predict(X_pred)
    predictions = np.argmax(predictions, axis=1)
    print(predictions)
    df = pd.DataFrame(list(zip(img_name, [class_names[pred] for pred in predictions], predictions)), columns=['img', 'class name', 'category'])
    print(df)
    if save_to_file:
        df.to_csv('../report/data/conv_predictions.csv')
