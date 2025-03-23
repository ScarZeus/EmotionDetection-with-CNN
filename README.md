# Emotion Detection CNN

This project uses a Convolutional Neural Network (CNN) to detect emotions from images. The dataset is organized into training and testing folders, each with subfolders for different emotion classes.

## Project Structure

- **src/**: Contains the Python modules.
  - **data_loader.py**: Loads and preprocesses the data.
  - **model.py**: Builds the CNN model.
  - **train.py**: Contains training, evaluation, and plotting functions.
  - **main.py**: The main script to run the project.
- **dataset/**: Contains `train/` and `test/` folders with images.
- **readme.md**: Project instructions.
- **requirements.txt**: Required Python libraries.

## How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the main script:
    ```bash
    python src/main.py
    ```
