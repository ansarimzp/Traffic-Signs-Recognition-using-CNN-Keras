# Traffic-Signs-Recognition-using-CNN-Keras

A web application built with Flask and TensorFlow/Keras that classifies German traffic signs from user-uploaded images. The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## Features

*   **Web Interface:** Simple frontend (likely defined in `templates/index.html`) for uploading traffic sign images [1].
*   **Image Classification:** Predicts the traffic sign category from the uploaded image using a trained Convolutional Neural Network (CNN) [1].
*   **43 Classes:** Recognizes 43 different types of German traffic signs [1][3].
*   **Confidence Score:** Provides the confidence level of the prediction [1].
*   **API Endpoint:** Includes a `/predict` endpoint for programmatic access [1].

## Dataset

This project utilizes the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset [3].
*   Contains images of 43 different traffic sign classes [3].
*   Images are preprocessed to 30x30 pixels with 3 color channels (RGB) for model input [3][1].

## Model Architecture

The classification model (`traffic_sign_model.h5`) is a Convolutional Neural Network (CNN) built using TensorFlow/Keras. The architecture, based on the training notebook (`gtsrb-cnn.ipynb`), includes [3]:
*   Multiple `Conv2D` layers with ReLU activations for feature extraction.
*   `MaxPool2D` layers for down-sampling.
*   `BatchNormalization` layers for stabilizing training.
*   A `Flatten` layer to prepare features for dense layers.
*   `Dense` layers with ReLU activation.
*   `Dropout` for regularization to prevent overfitting.
*   A final `Dense` layer with 43 units and `softmax` activation for multi-class classification [3].

*(Note: A PyTorch model definition (`model.py`) is also present, but the deployed application (`app.py`) uses the TensorFlow/Keras model saved in `traffic_sign_model.h5`)* [1][2][3].

## Technology Stack

*   **Backend:** Python, Flask [1]
*   **Machine Learning:** TensorFlow/Keras [1][3]
*   **Libraries:** NumPy, Pillow (PIL), Flask-CORS [1][3]
*   **(Training/Analysis):** Pandas, Matplotlib, Scikit-learn, Seaborn, Jupyter Notebook [3]

## Installation

1.  **Clone the repository:**
    ```
    git clone <(https://github.com/ansarimzp/Traffic-Signs-Recognition-using-CNN-Keras)>
    cd <REPOSITORY_DIRECTORY>
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    *   Ensure you have Python 3.x installed.
    *   Create a `requirements.txt` file with the following content:
        ```
        Flask
        Flask-Cors
        numpy
        tensorflow
        Pillow
        # Add other dependencies if needed (e.g., gunicorn for deployment)
        ```
    *   Install the requirements:
        ```
        pip install -r requirements.txt
        ```
4.  **Ensure Model File:** Make sure the `traffic_sign_model.h5` file is in the same directory as `app.py`, or update the path in `app.py` accordingly [1].

## Usage

1.  **Run the Flask application:**
    ```
    python app.py
    ```
2.  **Access the web interface:** Open your web browser and navigate to `http://127.0.0.1:5000/` (or the address provided in the console output) [1].
3.  **Upload an image:** Use the file upload form to select a traffic sign image and submit it.
4.  **View prediction:** The application will display the predicted traffic sign label and the confidence score [1].

**API Usage:**

You can also send a POST request directly to the `/predict` endpoint with the image file:

