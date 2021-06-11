import logging
import os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Silence tensorflow a bit

import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score



cmap = plt.cm.get_cmap('Paired')
col_true = cmap(1)
col_false = cmap(5)

sns.set_style("whitegrid")
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

def plot_image(i, predictions_array, true_label, img, class_names):
    """ Plot an image and sets the x-label
    to the predicted class.

    Args:
    i (int) :: index to select an item from
                prediction_array, true_label and img
    predictions_array (array) :: predicted labels
    true_label (array) :: true labels
    img (array) :: image to be plotted
    """
    assert (isinstance(i, int))
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if (predicted_label == true_label):
        color = col_true
    else:
        color = col_false

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    """ Create a bar plot of the predictions.

    Args:
    i (int) :: index to select an item from
               predictions_array and true_label
    predictions_array (array) :: predicted labels
    true_labels (array) :: true labels
    """
    assert (isinstance(i, int))
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color(col_false)
    thisplot[true_label].set_color(col_true)


def create_model():
    """
    Function to create a convolutional model and compiles it.
    :return: Returns the keras model.
    """
    model = keras.Sequential([
        keras.layers.Conv2D(filters=10, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(64, 64, 3)),
        keras.layers.Conv2D(filters=10, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(64, 64, 3)),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(6, activation="softmax")
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save_history(history):
    # summarize history for accuracy
    # fig, ax = plt.subplots(1,2, figsize=(14, 6))
    # residuals.plot(title="Residuals", ax=ax[0], legend=False)
    # residuals.plot(kind='kde', title='Density', ax=ax[1], legend=False)
    # plt.tight_layout()
    # plt.savefig('../report/images/res_dens.png')
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(1,2, figsize=(12, 5))
    ax[0].plot(history['accuracy'], marker='o')
    ax[0].plot(history['val_accuracy'], marker='o')
    ax[0].set_title('model accuracy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'validation'],)

    # summarize history for loss
    ax[1].plot(history['loss'], marker='o')
    ax[1].plot(history['val_loss'], marker='o')
    ax[1].set_title('model loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'validation'])
    plt.tight_layout()
    plt.savefig('../report/images/history.png')
