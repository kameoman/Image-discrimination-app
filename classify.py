import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Nadam  # Stochastic Gradient Descend
from keras.utils import np_utils


st.write("""
# Neural Network with 1 layer 
""")

np.random.seed(1671)  # for reproducibility

# network and training
epochs = st.number_input('Fill in number of epoch: ', min_value=12, format=None, step=1)
batch_sz = st.number_input('Specify batch size: ', min_value=168, format=None, step=8)
amount_of_output_classes = st.number_input('Specify amount of output classes', min_value=10, format=None, step=1)
NB_EPOCH = epochs
BATCH_SIZE = batch_sz
VERBOSE = 1
NB_CLASSES = amount_of_output_classes  # number of outputs = number of digits
amount_of_hidden_nodes = st.number_input('Specify amount of hidden nodes', min_value=168, format=None, step=1)
N_HIDDEN = amount_of_hidden_nodes
st.write("""
#### Optimizer Stochastic Gradient Descent
""")
OPTIMIZER = Nadam()  # SGD optimizer, explained later in this chapter
VALIDATION_SPLIT = 0.2  # how much TRAIN is reserved for VALIDATION

# data: shuffled and split between train and test sets
#
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784
#
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize 
#
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 10 outputs
# final stage is softmax

model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
activation_func_name = st.radio("Activation function name: ", ('softmax', 'relu', 'sigmoid', 'tanh'))
model.add(Activation(activation_func_name))

model.summary()

loss_func_name = st.radio("Loss function name: ",
                          ('categorical_crossentropy', 'kullback_leibler_divergence', 'poisson'))
model.compile(loss=loss_func_name,
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH,
                    verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT)
# list all data in history
print(history.history.keys())


# plot for accuracy train and test
fig, ax = plt.subplots()
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')
line1 = ax.plot(range(1, epochs + 1), history.history['accuracy'], label='train')
line2 = ax.plot(range(1, epochs + 1), history.history['val_accuracy'], label='test')
ax.legend(['train', 'test'])
st.pyplot(fig)

# plot for loss train and test
fig, ax = plt.subplots()
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')
line1 = ax.plot(range(1, epochs + 1), history.history['loss'], label='train')
line2 = ax.plot(range(1, epochs + 1), history.history['val_loss'], label='test')
ax.legend(['train', 'test'])
st.pyplot(fig)




score = model.evaluate(X_test, Y_test, verbose=VERBOSE)

md_results = f"Test loss is **{score[0]}** \n" \
             f" Test accuracy: **{score[1]}**."
st.markdown(md_results)

print("\nTest loss value :", score[0])
print('Test accuracy :', score[1])