{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GrainPalette – Rice Type Classification Using Deep Learning\n",
    "## A Deep Learning Odyssey using MobileNetV2 and Transfer Learning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Introduction\n",
    "This project classifies five types of rice grains using transfer learning with MobileNetV2.\n",
    "It demonstrates the power of CNNs in agricultural image classification using simulated image data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required libraries\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "from tensorflow.keras.applications import MobileNetV2\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.layers import Dense, GlobalAveragePooling2D\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from sklearn.model_selection import train_test_split\n",
    "from PIL import Image"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Simulate a dummy rice dataset (for demo only)\n",
    "We generate synthetic image data since no real dataset is included."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rice_types = ['Basmati', 'Jasmine', 'Arborio', 'Sona Masoori', 'Red Rice']\n",
    "\n",
    "def generate_dummy_data(samples_per_class=100):\n",
    "    X, y = [], []\n",
    "    for i, label in enumerate(rice_types):\n",
    "        for _ in range(samples_per_class):\n",
    "            # Dummy image: random pixels\n",
    "            img = np.random.rand(224, 224, 3)\n",
    "            X.append(img)\n",
    "            y.append(i)\n",
    "    return np.array(X), to_categorical(y, num_classes=len(rice_types))\n",
    "\n",
    "X, y = generate_dummy_data()\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Build the Rice Classifier Model with MobileNetV2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))\n",
    "x = base_model.output\n",
    "x = GlobalAveragePooling2D()(x)\n",
    "x = Dense(128, activation='relu')(x)\n",
    "predictions = Dense(len(rice_types), activation='softmax')(x)\n",
    "\n",
    "model = Model(inputs=base_model.input, outputs=predictions)\n",
    "for layer in base_model.layers:\n",
    "    layer.trainable = False\n",
    "\n",
    "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Train the model (demo training)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=32)\n",
    "\n",
    "# Optional: plot training accuracy and loss\n",
    "plt.figure(figsize=(10,4))\n",
    "plt.subplot(1,2,1)\n",
    "plt.plot(history.history['accuracy'], label='train_acc')\n",
    "plt.plot(history.history['val_accuracy'], label='val_acc')\n",
    "plt.title('Accuracy')\n",
    "plt.legend()\n",
    "\n",
    "plt.subplot(1,2,2)\n",
    "plt.plot(history.history['loss'], label='train_loss')\n",
    "plt.plot(history.history['val_loss'], label='val_loss')\n",
    "plt.title('Loss')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Upload an image and get prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    from google.colab import files\n",
    "    uploaded = files.upload()\n",
    "\n",
    "    for file_name in uploaded.keys():\n",
    "        img = Image.open(file_name).resize((224, 224)).convert('RGB')\n",
    "        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)\n",
    "        prediction = model.predict(img_array)[0]\n",
    "        predicted_index = np.argmax(prediction)\n",
    "        confidence = round(100 * prediction[predicted_index], 2)\n",
    "        print(f\"Predicted: {rice_types[predicted_index]} ({confidence}%)\")\n",
    "        plt.imshow(img)\n",
    "        plt.title(f\"{rice_types[predicted_index]} ({confidence}%)\")\n",
    "        plt.axis('off')\n",
    "        plt.show()\n",
    "except ImportError:\n",
    "    print(\"Not running in Colab. Please place test images in the working directory.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Save the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('mobilenet_rice_model.h5')"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "name": "GrainPalette_Rice_Classifier.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
