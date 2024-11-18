# iHofman

## Directory Structure

    ├── data_utils.py               # Data processing module
    ├── feature_extraction.py       # Feature generation module
    ├── model_utils.py              # Model creation module (autoencoder and attention)
    ├── train_and_evaluate.py       # Model training and evaluation module
    ├── main.py                     # Main program entry
    ├── README.md                   # Project documentation
    └── database/                   # Folder to store input data files
    
## Running the Main Program
To run the program, simply execute the main.py file. This will initiate the full process of data loading, feature extraction, model training, and evaluation.

    python main.py
    
    
## Modules Description
### 1. Data Processing Module (data_utils.py)
This module handles reading and storing data from CSV files. It provides functions to read data into memory and save processed data back into CSV files.

### 2. Feature Generation Module (feature_extraction.py)
This module is responsible for generating various types of features, including embedding features, sample features, and behavior features.

### 3. Model Creation Module (model_utils.py)
This module defines the deep learning models used in the project, including the stacked autoencoder (SAE) and attention mechanism models.

### 4. Model Training and Evaluation Module (train_and_evaluate.py)
This module handles the training of autoencoder models, evaluation of classifiers, and prediction outputs.

### 5. Main Program (main.py)
The main program integrates all modules and executes the full workflow. It processes data, generates features, trains the models, and evaluates their performance using different classifiers.


### Requirements
Ensure that the following Python packages are installed:

    pip install numpy keras scikit-learn joblib tqdm matplotlib


























