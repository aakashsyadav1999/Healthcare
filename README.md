# Healthcare

About Dataset
Context:
This synthetic healthcare dataset has been created to serve as a valuable resource for data science, machine learning, and data analysis enthusiasts. It is designed to mimic real-world healthcare data, enabling users to practice, develop, and showcase their data manipulation and analysis skills in the context of the healthcare industry.

Inspiration:
The inspiration behind this dataset is rooted in the need for practical and diverse healthcare data for educational and research purposes. Healthcare data is often sensitive and subject to privacy regulations, making it challenging to access for learning and experimentation. To address this gap, I have leveraged Python's Faker library to generate a dataset that mirrors the structure and attributes commonly found in healthcare records. By providing this synthetic data, I hope to foster innovation, learning, and knowledge sharing in the healthcare analytics domain.

Dataset Information:
Each column provides specific information about the patient, their admission, and the healthcare services provided, making this dataset suitable for various data analysis and modeling tasks in the healthcare domain. Here's a brief explanation of each column in the dataset -

Name: This column represents the name of the patient associated with the healthcare record.
Age: The age of the patient at the time of admission, expressed in years.
Gender: Indicates the gender of the patient, either "Male" or "Female."
Blood Type: The patient's blood type, which can be one of the common blood types (e.g., "A+", "O-", etc.).
Medical Condition: This column specifies the primary medical condition or diagnosis associated with the patient, such as "Diabetes," "Hypertension," "Asthma," and more.
Date of Admission: The date on which the patient was admitted to the healthcare facility.
Doctor: The name of the doctor responsible for the patient's care during their admission.
Hospital: Identifies the healthcare facility or hospital where the patient was admitted.
Insurance Provider: This column indicates the patient's insurance provider, which can be one of several options, including "Aetna," "Blue Cross," "Cigna," "UnitedHealthcare," and "Medicare."
Billing Amount: The amount of money billed for the patient's healthcare services during their admission. This is expressed as a floating-point number.
Room Number: The room number where the patient was accommodated during their admission.
Admission Type: Specifies the type of admission, which can be "Emergency," "Elective," or "Urgent," reflecting the circumstances of the admission.
Discharge Date: The date on which the patient was discharged from the healthcare facility, based on the admission date and a random number of days within a realistic range.
Medication: Identifies a medication prescribed or administered to the patient during their admission. Examples include "Aspirin," "Ibuprofen," "Penicillin," "Paracetamol," and "Lipitor."
Test Results: Describes the results of a medical test conducted during the patient's admission. Possible values include "Normal," "Abnormal," or "Inconclusive," indicating the outcome of the test.
Usage Scenarios:
This dataset can be utilized for a wide range of purposes, including:

Developing and testing healthcare predictive models.
Practicing data cleaning, transformation, and analysis techniques.
Creating data visualizations to gain insights into healthcare trends.
Learning and teaching data science and machine learning concepts in a healthcare context.
You can treat it as a Multi-Class Classification Problem and solve it for Test Results which contains 3 categories(Normal, Abnormal, and Inconclusive).
Acknowledgments:
I acknowledge the importance of healthcare data privacy and security and emphasize that this dataset is entirely synthetic. It does not contain any real patient information or violate any privacy regulations.
I hope that this dataset contributes to the advancement of data science and healthcare analytics and inspires new ideas. Feel free to explore, analyze, and share your findings with the Kaggle community.
Image Credit:
Image by BC Y from Pixabay


## Dataset

- Dataset Name: Healthcare
- Number of Records: 10000 entries, 0 to 9999
- Data Source: https://www.kaggle.com/datasets/prasad22/healthcare-dataset

## Approach
~~~
1. Data Exploration     : I started exploring dataset using pandas,numpy,matplotlib and seaborn. 

2. Data visualization   : Ploted graphs to get insights about dependend and independed variables. 

3. Feature Engineering  :  All The Value Are Arrange In One Range.

4. Model Selection I    :  Tested all base models to check the base accuracy.
                       
5. Model Selection II   :  Performed Hyperparameter tuning using gridsearchCV.

6. Pickle File          :  Selected model as per best accuracy and created pickle file.

7. Documentation        :  Created detailed document of the entire project, including data sources, preprocessing steps, model details, and results.
~~~


# Project Demo
Below providing the link of all the document that are required for creating the project
Link: [Document link](https://dagshub.com/aakashsyadav1999/Healthcare)

### Dags hub Experiments- ML FLow
```
MLFLOW_TRACKING_URI=https://dagshub.com/aakashsyadav1999/Healthcare.mlflow \
MLFLOW_TRACKING_USERNAME=aakashsyadav1999 \
MLFLOW_TRACKING_PASSWORD=2dea354c2b5d49805e93f9bc8b6cbdc23b7a516f \
python script.py
```

### Prediction Page

![Image]('D:\vscode\Healthcare\Docs\Images\Screenshot 2023-11-23 163827.png')


# Installation

To run my app on your local machine, do the following steps.

## Step 1 :

I have written the Code with Python 3.9.17. If you don't have Python installed you can find it here.
If you are using a lower version of Python you can upgrade using the pip package, kindly ensure that you have the latest version of pip.

## Step 2 :

If you want the current version of my repository to be in your github, you can do forking my repository visiting https://github.com/aakashsyadav1999/Healthcare

Clone my repository to your local machine by running the following command. Before doing this, you have to install git on your machine and make sure you are having proper internet connection.

For Windows OS user, open git bash and run the following command.

git clone https://github.com/aakashsyadav1999/Healthcare.git

For Linus OS user, open Terminal and run the following command.

git clone https://github.com/aakashsyadav1999/Healthcare.git

If you don't want to mess up with all these things, you can just download the zip file of my GitHub repository by clicking here and extract it to any file location as your wish and then use it.

Now we have done with the downloading of my whole project.

## Step 3 :

After downloading the whole repo, get into the main folder by hit the following command in git bash for Windows OS users and Terminal for Linux OS users.

cd Healthcare/

## Step 4 :

Now we are going to install all the dependency libraries for this project. Before that you must have Python 3.9.17 and latest version of pip.

To install all the dependency libraries in a single command, run the following command.

pip install -r requirements.txt

## Step 5 :

After installing all the dependency libraries, you are ready to run my app on your local machine.

To launch my app on your local machine, hit the following command.

python app.py

## Run

Now you have successfully launched my app on your local machine.

To view my app, hit the following URL in any of the browser such as Chrome, FireFox, etc..,

## http://127.0.0.1:5000 - For welcome page

## http://127.0.0.1:5000/predictdata - for prediction site


# Contributer
- Aakash Yadav