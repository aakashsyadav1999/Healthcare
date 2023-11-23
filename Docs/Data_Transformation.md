# Data Transformation Process

## Introduction

This document outlines the data transformation process implemented in the code, which involves various operations such as feature extraction, outlier removal, feature selection, feature encoding, and scaling. The primary objective is to prepare the data for machine learning tasks.

## Code Structure

The data transformation process is encapsulated in the `DataTransformation` class and is defined by the following components:

- **DataTransformationConfig**: A data class that stores the file path for the preprocessor object.

- Methods for feature extraction, outlier removal, feature selection, feature encoding, oversampling, and standardization.

## Data Transformation Steps

### 1. Feature Extraction

The `feature_extraction` method creates a new feature, `total_stays`, by adding the values of `stays_in_weekend_nights` and `stays_in_week_nights`. This new feature can provide additional insights for analysis.

### 2. Outlier Removal

The `outlier_removal` method removes outliers from the dataset. Outliers are detected and filtered based on selected numerical columns. The Interquartile Range (IQR) method is used for outlier detection.

### 3. Feature Selection

The `feature_selection` method removes non-essential features from the dataset. Several columns, including demographic and date-related information, are dropped to focus on relevant features.

### 4. Feature Encoding

The `features_encoding` method encodes categorical columns using Target Encoding. Target Encoding is applied to both training and testing datasets separately, and encoded data is obtained for further processing.

### 5. Oversampling (SMOTE)

The `oversampling_smote` method applies the Synthetic Minority Over-sampling Technique (SMOTE) to address imbalanced data in the training dataset. SMOTE generates synthetic samples to balance the class distribution.

### 6. Standardization

The `standardization` method performs feature scaling on the data using StandardScaler. This ensures that all features have a mean of 0 and a standard deviation of 1.

### 7. Preprocessing Object

The `get_data_transformation_object` method builds a preprocessing object, which includes encoding and scaling pipelines for both numerical and categorical features.

### 8. Data Transformation

The `initiate_data_transformation` method orchestrates the entire data transformation process. It involves reading the training and testing data, applying the above steps, and saving the preprocessing object.

## Usage

To use the data transformation process, follow these steps:

1. Create an instance of the `DataTransformation` class.
2. Call the `initiate_data_transformation(train_path, test_path)` method on the instance, providing the file paths to the training and testing data CSV files.
3. The method returns preprocessed data arrays and saves the preprocessing object to the specified file path.

## Conclusion

The data transformation process described in this document is a critical step in preparing data for machine learning and analysis. It involves feature engineering, data cleaning, and encoding to ensure the data is suitable for training machine learning models.
