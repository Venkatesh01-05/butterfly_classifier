🦋 Butterfly Classifier using MobileNetV2 (Google Colab)
This project implements a butterfly species image classifier using MobileNetV2 and TensorFlow. It's built and trained in Google Colab, leveraging a custom dataset of butterfly images.

📁 Dataset
The dataset is structured into train, val, and test folders with subfolders representing classes: butterfly_dataset_sample/ └── butterfly_dataset_sample/ ├── train/ │ ├── monarch/ │ └── swallowtail/ ├── val/ │ ├── monarch/ │ └── swallowtail/ └── test/ ├── monarch/ └── swallowtail/

📦 Sample ZIP: butterfly_dataset_sample (1).zip
You can upload this ZIP to your Google Drive and the script handles extraction.

🚀 Features
Transfer learning with MobileNetV2 pretrained on ImageNet
Data augmentation via ImageDataGenerator
Training and validation performance visualization
Confusion matrix analysis
Supports custom image prediction upload in Colab
Model saving to Google Drive
🧠 Model Summary
Base: MobileNetV2 (frozen)
Layers: GlobalAveragePooling2D → Dropout(0.3) → Dense(softmax)
Optimizer: Adam
Loss: categorical_crossentropy
📒 Usage
Open Google Colab
Mount Google Drive and upload:
butterfly_dataset_sample (1).zip
butterfly_classifier_colab.py (this script)
Run the entire script.
📦 Output
Trained model: butterfly_classifier_mobilenetv2.h5 saved to Google Drive
Final evaluation: Accuracy + Loss + Confusion Matrix
Live prediction from uploaded image (JPG/PNG)
🔖 License
This project is licensed under the MIT License.

Feel free to improve the model architecture or train on more classes!

