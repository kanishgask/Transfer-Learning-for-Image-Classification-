# Transfer-Learning-for-Image-Classification-
📦 CIFAR-10 Classification with Transfer Learning (MobileNetV2)
This project demonstrates image classification on the CIFAR-10 dataset using Transfer Learning with a pre-trained MobileNetV2 model from Keras.

**📌 Overview**
Leverages MobileNetV2 pretrained on ImageNet

Adapts it to classify CIFAR-10's 10 object categories

Freezes the base model and trains a custom classification head

Visualizes prediction results

Saves the final model for reuse

**🧰 Requirements**
Python 3.x

TensorFlow 2.x

NumPy

Matplotlib

Install required libraries:

bash
Copy
Edit
pip install tensorflow numpy matplotlib
**🧠 Model Architecture**
Base: MobileNetV2 without the top classifier (include_top=False)

Input resized to 96x96 (original MobileNetV2 input size)

Added:

GlobalAveragePooling2D

Dense(128, activation='relu')

Dense(10, activation='softmax') for CIFAR-10 classes

**🗂️ Dataset**
CIFAR-10: 60,000 32x32 color images in 10 classes, with 6,000 images per class.

Automatically downloaded using keras.datasets.cifar10.

🛠️ Steps
1. Load and Normalize Data
Data loaded using Keras datasets

Pixel values scaled to [0, 1]

Labels one-hot encoded

2. Resize Images
CIFAR-10 images resized from 32x32 to 96x96 to match MobileNetV2 input

3. Build Model
Use pretrained MobileNetV2 as feature extractor

Add custom dense layers for classification

4. Train Model
Base model layers are frozen (weights not updated)

Custom classifier layers are trained for 5 epochs

5. Evaluate and Visualize
Final accuracy is printed

Function predict_sample(index) shows prediction for a test image

**📈 Results**
Example output:

yaml
Copy
Edit
Epoch 5/5
782/782 [==============================] - 20s 25ms/step - loss: 0.7638 - accuracy: 0.7462 - val_loss: 0.8753 - val_accuracy: 0.7114
Test Accuracy: 0.71
**Visualization example:**

python
Copy
Edit
predict_sample(7)
Displays an image from the test set along with predicted and actual class labels.

🧪 Prediction Function
python
Copy
Edit
def predict_sample(index):
    sample = X_test[index].numpy().reshape(1, 96, 96, 3)
    prediction = model.predict(sample)
    pred_class = class_names[np.argmax(prediction)]
    actual_class = class_names[np.argmax(y_test[index])]
    plt.imshow(X_test[index].numpy())
    plt.title(f"Predicted: {pred_class} | Actual: {actual_class}")
    plt.axis('off')
    plt.show()
**💾 Model Saving**
After training:

python
Copy
Edit
model.save("transfer_learning_cifar10_model.h5")
You can later load the model using:

python
Copy
Edit
from tensorflow.keras.models import load_model
model = load_model("transfer_learning_cifar10_model.h5")
**📚 Classes**
css
Copy
Edit
['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
**📝 Notes**
The base MobileNetV2 layers are frozen to retain pretrained features

Only the top classifier is trained on CIFAR-10

For better results, consider unfreezing top layers and fine-tuning

**📄 License**
This project is licensed under the MIT License.
