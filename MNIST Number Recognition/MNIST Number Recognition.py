{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "bfa98d8b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "0b659bb8",
   "metadata": {},
   "outputs": [],
   "source": [
    "mnist= tf.keras.datasets.mnist  #using builtin dataset of mnist inside keras"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "64fcef17",
   "metadata": {},
   "outputs": [],
   "source": [
    "(train_data, train_label), (test_data, test_label)= mnist.load_data() #download mnist data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "3e229b5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "neural= tf.keras.Sequential(\n",
    "    [tf.keras.layers.Flatten(input_shape=(28,28)), #flattening defaulf pixel size/ Input layer 1, 784 nodes\n",
    "     tf.keras.layers.Dense(128,activation=tf.nn.relu), #hiddden layer 1, 128 nodes, multiple HLs possible\n",
    "     tf.keras.layers.Dense(10,activation=tf.nn.softmax) #output layer, 10 nodes(possible outputs numbers 0-9 )\n",
    "    ])  #using softmax at output to get predictions strictly between 0 & 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "ee9bb8c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "#loss function\n",
    "neural.compile(\n",
    "    optimizer = tf.keras.optimizers.Adam(),\n",
    "    loss = 'sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy']) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "b31c5c07",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/6\n",
      "1875/1875 [==============================] - 9s 4ms/step - loss: 2.5519 - accuracy: 0.8593\n",
      "Epoch 2/6\n",
      "1875/1875 [==============================] - 9s 5ms/step - loss: 0.3779 - accuracy: 0.9077\n",
      "Epoch 3/6\n",
      "1875/1875 [==============================] - 8s 4ms/step - loss: 0.2916 - accuracy: 0.9254\n",
      "Epoch 4/6\n",
      "1875/1875 [==============================] - 8s 4ms/step - loss: 0.2519 - accuracy: 0.9352\n",
      "Epoch 5/6\n",
      "1875/1875 [==============================] - 8s 4ms/step - loss: 0.2289 - accuracy: 0.9410\n",
      "Epoch 6/6\n",
      "1875/1875 [==============================] - 7s 4ms/step - loss: 0.2099 - accuracy: 0.9463\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x2110c367a30>"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "neural.fit(train_data, train_label, epochs=6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "6cc5f4c7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 1s 3ms/step - loss: 0.2829 - accuracy: 0.9408\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[0.2829079031944275, 0.9408000111579895]"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#test on testing data\n",
    "neural.evaluate(test_data, test_label) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "d3ba5a4e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 1s 2ms/step\n",
      "7\n",
      "[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]\n"
     ]
    }
   ],
   "source": [
    "#randomized test on testing data\n",
    "random= neural.predict(test_data)\n",
    "np.set_printoptions(suppress=True)\n",
    "print(test_label[0])\n",
    "print(random[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5a5cdc2d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Ashutosh Mahajan"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
