### Full and extensive analysis can be found in the file eda.py.

- [I. PLANNING](#I)
    - [I.1 Introduction](#I.1)
    - [I.2 Dataset description](#I.2)
    - [I.3 Project assumptions](#I.3)
        - [I.3.1 Defining the problem](#I.3.1)
        - [I.3.2 Assessing the scope](#I.3.2)
        - [I.3.3 Success metric](#I.3.3)
        - [I.3.4 Feasibility  of the ML application](#I.3.4)
- [II.DATA COLLECTION AND PREPARATION](#II)
    - [II.1 Import libraries and data files](#II.1)
    - [II.2 Exploratory data analysis (EDA)](#II.2)
        - [II.2.1 Reading data & target C=class distribution](#II.2.1)
        - [II.2.2 Statistical summary](#II.2.2)
        - [II.2.3 Correlation Matrics](#II.2.3)
        - [II.2.4 Distribution of attributes with fewer than 10 unique values](#II.2.4)
        - [II.2.5 Distribution of attributes with fewer than 10 unique values](#II.2.5)
            - [II.2.5.1 Distribution of CODE_GENDER](#II.2.5.1)
            - [II.2.5.2 Distribution of FLAG_OWN_CAR](#II.2.5.2)
            - [II.2.5.3 Distribution of FLAG_OWN_REALTY](#II.2.5.3)
            - [II.2.5.4 Distribution of NAME_TYPE_SUITE](#II.2.5.4)            
            - [II.2.5.5 Distribution of NAME_INCOME_TYPE](#II.2.5.5)            
            - [II.2.5.6 Distribution of NAME_EDUCATION_TYPE](#II.2.5.6)           
            - [II.2.5.7 Distribution of NAME_FAMILY_STATUS](#II.2.5.7)
            - [II.2.5.8 Distribution of NAME_HOUSING_TYPE](#II.2.5.8)
            - [II.2.5.9 Distribution of REGION_RATING_CLIENT](#II.2.5.9)            
            - [II.2.5.10 Distribution of WEEKDAY_APPR_PROCESS_START](#II.2.5.10)
            - [II.2.5.11 Distribution of ORGANIZATION_TYPE](#II.2.5.11)            
            - [III.2.5.12 Distribution of HOUSETYPE_MODE](#II.2.5.12)            
            - [II.2.5.13 Distribution of EMERGENCYSTATE_MODE](#II.2.5.13)    
            - [II.2.5.14 Distribution of NAME_CONTRACT_TYPE](#II.2.5.14) 
            - [II.2.5.15 Distribution of OCCUPATION_TYPE](#II.2.5.15) 
       - [II.2.6 Distribution of Gender](#II.2.6) 
       - [II.2.7 Distribution of AMT_CREDIT](#II.2.7)
       - [I.2.8  AMT_GOODS_PRICE](#II.2.8)       
       - [II.2.9 AMT_CREDIT](#II.2.9)     
       - [II.2.10 Didtibution of categorical](#II.2.10)       
       - [II.2.11 Name Contract Status in Previous Applications](#II.2.11)       
       - [II.2.12 Client Type in Previous Applications](#II.2.12) 
       - [II.2.13 Channel Type in Previous Applications](#II.2.13)     
- [III DATA PRE-PROCESSING (data cleaning)](#III)     
    - [III.1 Filling nulls](#IV.1)
    - [III.2 Feature Engineering](#IV.2) 
    - [III.3 Convert types (Downcasting)](#III.3)
    - [III.4 Skewness of distributions](#III.4)
    - [III.5 Detect outlier](#III.5)
    - [III.6 Categorical data transformation](#III.6)   
    - [III.7 Normalizing](#III.7)     
    - [III.8 TSN](#III.8)
    - [III.9 PCA](#III.9)
    - [III.10 Imbalanced target - oversampling by SMOTEE](#III.10)
    - [III.11 Feature selection](#III.11)
- [IV DATA PROCESSING](#IV)
- [V MODEL ENGINEERING](#V)
    - [V.1 XGBClassifier](#V.1)
        - [V.1.1 XGBClassifier - Evaluation](#V.1.1)
        - [V.1.2 XGBClassifier Tuning - RandomizedSearchCV](#V.1.2)
    - [V.2 LGBMClassifier Tuning - Optuna](#V.2) 
        - [V.2.1 LGBMClassifier - Threshold=0.2](#V.2.1)
    - [V.3 RandomForestClassifier](#V.3)
        - [V.3.1 RandomForestClassifier - TunedThresholdClassifierCV](#V.3.1)       
    - [V.4 SMOTE Classifierss](#V.4)   
        - [V.4.1 RandomForestClassifier - SMOTE](#V.4.1)
        - [V.4.2 XGBoostClassifier - SMOTE](#V.4.3)
- [VI CONCLUSION](#VI) 
   
   
I.1 Introduction
Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, the organizer requires to unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

I.2 Dataset description
Full feature description is in eda.py file

I.3 Project assumptions
I.3.1 Defining the problem
This project will be implemented based on a real dataset, provided by the project organizer. The objective of this problem is to use historical loan application data to predict whether or not an applicant will be able to repay a loan. This is a standard supervised classification task.

I.3.2 Assessing the scope
The entire project was done in Python, using Jupyter. Defining the scope of the project, its purpose and priorities determines the type of approach to it. In this case, the main goal is to achieve a predictive model result that exceeds the satisfaction value achieved by the organizer. We need to use all the data, for now we will stick to one file which should be more manageable. This will let us establish a baseline that we can then improve upon. With these projects, it's best to build up an understanding of the problem a little at a time rather than diving all the way in and getting completely lost.




### Author Details:
- Name: Adam Smulczyk
- Email: adam.smulczyk@gmail.com
- Profile: [Github](https://github.com/AdamSmulczyk)
- [Github Repository](https://github.com/AdamSmulczyk/014_Poisonous_Mushrooms)
