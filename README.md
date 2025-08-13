Image Classification Project with Keras
This project is a simple image classification system built using the Keras library with a VGG16 pre-trained model. The model is fine-tuned to classify images into specific categories, making it a powerful tool for visual recognition tasks.

Key Features
VGG16 Pre-trained Model: The project utilizes the VGG16 model, pre-trained on the ImageNet dataset, as a base. This allows for a fast and efficient training process on new datasets through transfer learning.

Simple Image Classification: The system is fine-tuned for a specific task, making it highly accurate at distinguishing between the classes present in the training data.

Efficient Workflow: The training script handles the entire training process, including data augmentation and saving the resulting model.

Prediction: A separate prediction script allows for easy testing of the model on new images.

Prerequisites
To run this project, you need to have Python installed. It's recommended to set up a virtual environment and install the required libraries. You can use the following command to install all dependencies:

pip install keras tensorflow numpy matplotlib

Getting Started
1. Clone the Repository
Clone this repository to your local machine using the following command:

git clone [Your Repository URL]

2. Install Dependencies
Navigate into the project directory and install the required libraries.

cd [Your Project Directory]
pip install -r requirements.txt

3. Train the Model
To train the model, you must provide your own dataset of images organized into separate folders for each class. The Training.py script will automatically load this data, fine-tune the VGG16 model, and save the resulting model file (final_model.h5) to a new directory named model.

Run the training script with this command:

python Training.py

4. Make Predictions
After training is complete, the predict.py script can be used to test the model on new images. The script will load the saved model and classify a new image based on its path.

Run the prediction script with this command:

python predict.py
