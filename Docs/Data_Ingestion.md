# Data Ingestion Process

## Introduction

This document provides an overview of the data ingestion process implemented in the code, which involves reading data from an Excel file, splitting it into training and testing sets, and saving the data in CSV format.

## Code Structure

The data ingestion process is encapsulated in the `DataIngestion` class and is defined by the following components:

- **DataIngestionConfig**: A data class that stores the file paths for the raw data, training data, and testing data.

- **initiate_data_ingestion()**: A method that performs the data ingestion process.

## Data Ingestion Steps

### 1. Reading Data

The data ingestion process begins by reading an Excel file named "bookings.xlsx." This dataset serves as the source data for subsequent processing. The data is loaded into a Pandas DataFrame.

### 2. Creating Directories

Before saving the data, the code creates directories to store the CSV files if they do not already exist. These directories are specified in the `DataIngestionConfig`.

### 3. Saving Raw Data

The raw data is saved to a CSV file specified by `raw_data_path`. This file contains the entire dataset as it was read from the Excel file, preserving its integrity.

### 4. Train-Test Split

A standard train-test split is performed on the dataset using the `train_test_split` function from the `sklearn.model_selection` library. The training set contains 80% of the data, while the testing set contains 20%. A random seed (42) is used for reproducibility.

### 5. Saving Training and Testing Data

Both the training and testing datasets are saved to separate CSV files specified by `train_data_path` and `test_data_path`, respectively. These CSV files are ready for use in machine learning models or other data processing tasks.

## Exception Handling

The code includes exception handling to capture and raise any exceptions that may occur during the data ingestion process. These exceptions are caught and converted into a custom exception (`CustomException`) along with relevant information such as the error message and system details.

## Logging

Logging statements are included in the code to provide insights into the execution of the data ingestion process. Key events and stages in the process are logged to assist in debugging and monitoring.

## Usage

To use the data ingestion process, follow these steps:

1. Create an instance of the `DataIngestion` class.
2. Call the `initiate_data_ingestion()` method on the instance to perform data ingestion.
3. The method returns the file paths for the training and testing data CSV files, which can be used for further analysis and modeling.

## Conclusion

The data ingestion process detailed in this document is a fundamental step in preparing data for machine learning and analysis. It provides clean, structured data in CSV format that can be readily used in various data science projects.