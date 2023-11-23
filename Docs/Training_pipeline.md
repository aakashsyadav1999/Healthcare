# Training Pipeline

## Introduction

This document focuses on both data ingestion and data transformation processes as part of the training pipeline. These processes are vital for gathering, preparing, and transforming the training data for subsequent machine learning model training. These processes are implemented in the `TrainingPipeline` class.

## Code Structure

The data ingestion and transformation processes are part of a larger training pipeline that includes model training and experiment tracking. The relevant components are as follows:

- **TrainingPipeline**: A class that orchestrates the entire training process, including data ingestion, transformation, model training, and experiment tracking.

- **DataIngestion**: A component responsible for loading and preparing the training and testing data.

- **DataTransformation**: A component responsible for feature extraction, outlier removal, feature selection, feature encoding, and data scaling.

## Data Ingestion Steps

### 1. Data Ingestion

The data ingestion process starts with the `DataIngestion` component. It performs the following steps:

- **Data Reading**: The code reads data from a source (in this case, "Notebook\data\healthcare_dataset.csv") using the Pandas library, creating a DataFrame.

- **Directory Creation**: It creates directories to store CSV files for raw data, training data, and testing data.

- **Data Splitting**: The data is split into training and testing sets using an 80/20 split ratio, and both sets are saved as CSV files in their respective directories.

## Data Transformation Steps

### 2. Data Transformation

The data transformation process is handled by the `DataTransformation` component and involves the following steps:

- **Feature Selection**: Non-essential features, such as demographic and date-related information, are removed from the dataset to focus on relevant features.

- **Feature Encoding**: Categorical columns are encoded using Target Encoding, separately for training and testing datasets.

- **Standardization**: Feature scaling is performed using StandardScaler to ensure that all features have a mean of 0 and a standard deviation of 1.

### 3. Training Pipeline

The `TrainingPipeline` class is responsible for orchestrating the entire training process. It includes the following steps:

- **Model Training**: The pipeline initiates model training using the data prepared by the `DataIngestion` and `DataTransformation` components.

- **Model Evaluation**: It evaluates the trained models, selecting the best-performing model based on various metrics.

- **Experiment Creation**: The pipeline creates an experiment using MLflow, logs the model, and associated metrics.

- **Remote Server Configuration (DAGShub)**: Optionally, it can be configured to work with remote servers like DAGShub for model tracking and registry.


## Usage

To use the training pipeline for model training and experiment tracking, follow these steps:

1. Create an instance of the `TrainingPipeline` class.

2. Call the `start_model_training` method on the instance to initiate the data ingestion and transformation, as well as model training and experiment tracking.

3. Specify the experiment name, run name, accuracy, precision, recall, best model, confusion matrix path, ROC AUC plot path, and run parameters if needed.

4. The pipeline orchestrates the process, trains the model, and logs relevant information to the MLflow tracking server or a remote server like DAGShub.

## Conclusion

The data ingestion and transformation processes are crucial initial steps in the training pipeline. They ensure that the training and testing data are correctly loaded, prepared, and transformed for subsequent model training and experiments. This document provides an overview of both processes within the larger training pipeline.

This document offers insights into the data ingestion and transformation processes within the broader context of the training pipeline, facilitating a better understanding of how data is prepared and transformed for model training and experiments.
