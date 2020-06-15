{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from keras import layers\n",
    "from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D\n",
    "from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D\n",
    "from keras.models import Model\n",
    "from keras.preprocessing import image\n",
    "from keras.utils import layer_utils\n",
    "from keras.utils.data_utils import get_file\n",
    "from keras.applications.imagenet_utils import preprocess_input\n",
    "import pydot\n",
    "from IPython.display import SVG\n",
    "from keras.utils.vis_utils import model_to_dot\n",
    "from keras.utils import plot_model\n",
    "from kt_utils import *\n",
    "\n",
    "import keras.backend as K\n",
    "K.set_image_data_format('channels_last')\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.pyplot import imshow\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of training examples = 600\n",
      "number of test examples = 150\n",
      "X_train shape: (600, 64, 64, 3)\n",
      "Y_train shape: (600, 1)\n",
      "X_test shape: (150, 64, 64, 3)\n",
      "Y_test shape: (150, 1)\n"
     ]
    }
   ],
   "source": [
    "X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()\n",
    "\n",
    "# Normalize image vectors\n",
    "X_train = X_train_orig/255.\n",
    "X_test = X_test_orig/255.\n",
    "\n",
    "# Reshape\n",
    "Y_train = Y_train_orig.T\n",
    "Y_test = Y_test_orig.T\n",
    "\n",
    "print (\"number of training examples = \" + str(X_train.shape[0]))\n",
    "print (\"number of test examples = \" + str(X_test.shape[0]))\n",
    "print (\"X_train shape: \" + str(X_train.shape))\n",
    "print (\"Y_train shape: \" + str(Y_train.shape))\n",
    "print (\"X_test shape: \" + str(X_test.shape))\n",
    "print (\"Y_test shape: \" + str(Y_test.shape))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: HappyModel\n",
    "\n",
    "def HappyModel(input_shape):\n",
    "    \"\"\"\n",
    "    Implementation of the HappyModel.\n",
    "    \n",
    "    Arguments:\n",
    "    input_shape -- shape of the images of the dataset\n",
    "        (height, width, channels) as a tuple.  \n",
    "        Note that this does not include the 'batch' as a dimension.\n",
    "        If you have a batch like 'X_train', \n",
    "        then you can provide the input_shape using\n",
    "        X_train.shape[1:]\n",
    "    \"\"\"\n",
    "\n",
    "    X_input=Input(input_shape)\n",
    "    \n",
    "    X=ZeroPadding2D((3,3))(X_input)\n",
    "    \n",
    "    X=BatchNormalization(axis=2)(X)\n",
    "    \n",
    "    X=Activation('relu')(X)\n",
    "    \n",
    "    X=MaxPooling2D((2,2))(X)\n",
    "    \n",
    "    X=Flatten()(X)\n",
    "    \n",
    "    X=Dense(1,activation='relu')(X)\n",
    "    \n",
    "    model=Model(inputs=X_input,outputs=X)\n",
    "    \n",
    "    #return model\n",
    "\n",
    "    \n",
    "    ### START CODE HERE ###\n",
    "    # Feel free to use the suggested outline in the text above to get started, and run through the whole\n",
    "    # exercise (including the later portions of this notebook) once. The come back also try out other\n",
    "    # network architectures as well. \n",
    "    \n",
    "    \n",
    "    ### END CODE HERE ###\n",
    "    \n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "### START CODE HERE ### (1 line)\n",
    "happyModel = HappyModel(X_train.shape[1:])\n",
    "### END CODE HERE ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "### START CODE HERE ### (1 line)\n",
    "opt=happyModel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])\n",
    "### END CODE HERE ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\HP\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\ops\\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use tf.where in 2.0, which has the same broadcast rule as np.where\n",
      "WARNING:tensorflow:From C:\\Users\\HP\\Anaconda3\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.\n",
      "\n",
      "Epoch 1/40\n",
      "600/600 [==============================] - 2s 4ms/step - loss: 6.7434 - accuracy: 0.1050\n",
      "Epoch 2/40\n",
      "600/600 [==============================] - 0s 753us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 3/40\n",
      "600/600 [==============================] - 0s 767us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 4/40\n",
      "600/600 [==============================] - 0s 813us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 5/40\n",
      "600/600 [==============================] - 1s 904us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 6/40\n",
      "600/600 [==============================] - 1s 865us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 7/40\n",
      "600/600 [==============================] - 0s 757us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 8/40\n",
      "600/600 [==============================] - 0s 741us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 9/40\n",
      "600/600 [==============================] - 0s 745us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 10/40\n",
      "600/600 [==============================] - 1s 853us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 11/40\n",
      "600/600 [==============================] - 1s 845us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 12/40\n",
      "600/600 [==============================] - 1s 851us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 13/40\n",
      "600/600 [==============================] - 0s 818us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 14/40\n",
      "600/600 [==============================] - 0s 805us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 15/40\n",
      "600/600 [==============================] - 1s 1ms/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 16/40\n",
      "600/600 [==============================] - 1s 847us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 17/40\n",
      "600/600 [==============================] - 0s 724us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 18/40\n",
      "600/600 [==============================] - 0s 743us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 19/40\n",
      "600/600 [==============================] - 0s 734us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 20/40\n",
      "600/600 [==============================] - ETA: 0s - loss: 7.7214 - accuracy: 0.0000e+ - 0s 726us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 21/40\n",
      "600/600 [==============================] - ETA: 0s - loss: 7.6932 - accuracy: 0.0000e+ - 0s 825us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 22/40\n",
      "600/600 [==============================] - 1s 874us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 23/40\n",
      "600/600 [==============================] - 1s 981us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 24/40\n",
      "600/600 [==============================] - 0s 795us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 25/40\n",
      "600/600 [==============================] - 0s 804us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 26/40\n",
      "600/600 [==============================] - 1s 1ms/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 27/40\n",
      "600/600 [==============================] - 1s 983us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 28/40\n",
      "600/600 [==============================] - 1s 917us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 29/40\n",
      "600/600 [==============================] - 1s 888us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 30/40\n",
      "600/600 [==============================] - 0s 765us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 31/40\n",
      "600/600 [==============================] - 0s 769us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 32/40\n",
      "600/600 [==============================] - 0s 748us/step - loss: 7.6666 - accuracy: 0.0000e+000s - loss: 8.4507 - accuracy\n",
      "Epoch 33/40\n",
      "600/600 [==============================] - 0s 765us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 34/40\n",
      "600/600 [==============================] - 0s 770us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 35/40\n",
      "600/600 [==============================] - 0s 750us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 36/40\n",
      "600/600 [==============================] - 0s 711us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 37/40\n",
      "600/600 [==============================] - 0s 714us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 38/40\n",
      "600/600 [==============================] - 0s 709us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 39/40\n",
      "600/600 [==============================] - 0s 736us/step - loss: 7.6666 - accuracy: 0.0000e+00\n",
      "Epoch 40/40\n",
      "600/600 [==============================] - 0s 701us/step - loss: 7.6666 - accuracy: 0.0000e+00\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.callbacks.History at 0x1cee12f4f98>"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### START CODE HERE ### (1 line)\n",
    "happyModel.fit(X_train,Y_train,epochs=40,batch_size=16)\n",
    "### END CODE HERE ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
