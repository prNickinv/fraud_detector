# Fraud Detection Service

By Nikita Artamonov

This repository contains a containerized microservice designed to score transactions for fraud. The service uses a [CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier) model to make predictions. The complete model training process is documented in the included [Jupyter Notebook](training/training.ipynb).

The application continuously monitors a designated `input` directory. When a new CSV file containing transaction data is added, the service automatically preprocesses the data, scores each transaction, and outputs the results, along with the top 5 most influential features values and predicted scores distribution plots, to an `output` directory.

## Features

-   **Automated Processing:** Monitors an `input` directory for new transaction data (`.csv` files) using `watchdog`.
-   **Machine Learning Model:** Scores transactions using a CatBoost classifier to predict whether a transaction is fraudulent (1) or not (0).
-   **Training:** Includes the `training.ipynb` notebook detailing the entire model training pipeline.
-   **Outputs:** For each processed file, the service generates:
    -   A `predictions_{timestamp_input}.csv` file with the prediction for each transaction.
    -   A `important_features.json` file detailing the top 5 most influential features values for each transaction.
    -   A `score_distribution.png` plot visualizing the distribution of the predicted fraud scores.
    -   A `predicted_labels_distribution.png` plot visualizing the distribution of the predicted fraud labels.
-   **Containerized:** Containerized with Docker.

## Project Structure

```
fraud_detector/
├── app/
│   └── app.py              # Main application script with the file watcher
├── input/                  # Directory for incoming transaction data (e.g., test.csv)
├── output/                 # Directory for generated predictions, important features and plots
├── logs/                   # Directory for service logs
├── models/
│   └── fraud_detector_model.cbm # CatBoost model
├── training/
│   └── training.ipynb      # Jupyter notebook showing the model training process
├── src/
│   ├── feature_importance.py
│   ├── plotting.py
│   ├── preprocessing.py
│   └── scorer.py
├── data/
    ├── train.csv
    ├── test.csv
├── config.py               # Config for paths and constants
├── Dockerfile              
├── requirements.txt     
├── .dockerignore
├── .gitignore   
└── README.md               
```

**Note:** the `data` directory is not included to this repository. The training data can be found on [Kaggle](https://www.kaggle.com/competitions/teta-ml-1-2025/data).

## Model Training

The `fraud_detector_model.cbm` used by the service was created using the Jupyter Notebook located at `training/training.ipynb`.

This notebook provides:
-   Data loading and preprocessing steps.
-   Training the `CatBoostClassifier` model.
-   Evaluating the model's performance.
-   Selecting a threshold for determing whether a transaction is fraudulent or not
-   Saving the model

## How to Launch the Service

Follow these steps to build the Docker image and run the service in a container.

### 1. Clone the Repository

Clone this repository and enter its directory with the following command:

```bash
git clone https://github.com/prNickinv/fraud_detector.git
cd fraud_detector
```

### 2. Build the Docker Image

From the root directory of the project (`fraud_detector/`), run the following command to build the Docker image. We will tag it as `fraud_detector`.

```bash
docker build -t fraud_detector .
```

### 4. Run the Docker Container

Execute the command below to start the service. This command runs the container and connects the local directories `input` and `output` to the directories inside the container via `-v` option.

```bash
docker run -it --rm -v ./input:/app/input \
                    -v ./output:/app/output \
                    fraud_detector
```

The service is now running and waiting for files.

### 5. Add Data

With the container running, drop a CSV file containing transaction data (e.g., `test.csv`) into the local `input` directory.

### 6. Get Results

Check the local `output` directory. You will find the generated files:

-   `predictions_{timestamp_input}.csv`
-   `important_features.json`
-   `score_distribution.png`
-   `predicted_labels_distribution.png`
