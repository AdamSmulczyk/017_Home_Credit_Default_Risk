### Full and extensive analysis can be found in the file eda.py.

* I. PLANNING
    * I.1 Introduction
    * I.2 Dataset description
    * I.3 Project assumptions
        * I.3.1 Defining the problem
        * I.3.2 Assessing the scope
        * I.3.3 Success metric
        * I.3.4 Feasibility of the ML application
* II.DATA COLLECTION AND PREPARATION
    * II.1 Import libraries and data files
    * II.2 Exploratory data analysis (EDA)
        * II.2.1 Reading data & target C=class distribution
        * II.2.2 Statistical summary
        * II.2.3 Correlation Matrics
        * II.2.4 Missing values, categorical data transformation
        * II.2.5 Distribution of attributes with fewer than 10 unique values
        * II.2.6 Distribution of numerical features
        * II.2.7 Distribution of categorical features
        * II.2.8 Distribution of target class in season
        * II.2.9 Correlations between Numerical Features
* III DATA PRE-PROCESSING
    * III.1 Target encoding
    * III.2 Filling nulls
    * III.3 Removing duplicates and unnecessary columns
    * III.4 Filling nulls
    * III.5 Filling nulls
    * III.6 Convert types (downcasting)
* IV DATA PROCESSING
    * IV.1 Skewness of distributions
    * IV.2 Detect outlier
    * IV.3 Categorical data transformation
    * IV.4 Normalizing
    * IV.5 TSN
    * IV.6 PCA
    * IV.7 Feature selection
    * IV.8 Imbalanced target - oversampling by SMOTEE
   
   
I.1 Introduction
The goal is to predict whether a mushroom is edible or poisonous based on its physical features, such as color, shape, and texture.

To tackle this problem, we'll be analyzing a special dataset. This dataset was created by a deep learning model that studied thousands of mushrooms. While the data is similar to a well-known mushroom dataset, there are some differences that make this project unique.

This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like leaflets three, let it be'' for Poisonous Oak and Ivy.

I.2 Dataset description
Full feature description is in eda.py file

I.3 Project assumptions
I.3.1 Defining the problem
This project will be implemented based on a real dataset, provided by the project organizer. The goal is to develop a model that can classify mushrooms as edible ('e') or poisonous ('p') using a set of physical attributes provided in the dataset.

I.3.2 Assessing the scope
The entire project was done in Python, using Jupyter. Defining the scope of the project, its purpose and priorities determines the type of approach to it. In this case, the main goal is to predict whether a mushroom is edible or poisonous based on its physical features, such as color, shape, and texture.

II.DATA COLLECTION AND PREPARATION
Conclusions:
* There is no duplicates in both files.
* There is 3116945 records but distribution is almost equal, e-45%, p-55%.
* With three million rows, the training dataset is huge. And with this large amount of data, we'll focus on gradient-boosted tree models and neural networks.
* Most features have missing values. We'll either use models which can deal with missing values natively, or we'll have to impute the missing values.
* Most features are categorical. We can experiment with one-hot encoding and target encoding.
* For numerical data skew is normal.
* Categorical data can't be dummies because nunique value for them is high.
* Positive Correlations:
    * 'stem-root' shows strong positive correlation with target class.
    * 'stem-width' shows strong positive correlation with 'cap-diameter'.
    * 'veil-color' shows strong positive correlation with 'stem-surface'.
    * 'cap-diameter', 'stem-surface','stem-width', 'stem-root' and 'stem-height' shows a moderate positive correlations with features.
* Negative Correlations:
    * 'veil-color' and 'spore-print-color' shows strongly negative correlation with target class.
    * Vehicle_Age and Policy_Sales_Channel status have a moderate negative correlation with Response.
    * Age status show strongly negative correlation with Policy_Sales_Channel, Vehicle_Age and Previously_Insured.
* We encounter a challenge: some categories don't show up very often in our data. This makes it hard to work with them. To fix this, we'll group these rare categories together into a new category called "Unknown".
* Distribution target class is almost equal.
* Cap-shape: although most of b and o type is poisoning.
* Distribution is very unbalanced: 'does-bruise-or-bleed', 'ring-type', 'habitat'.
* Gil-spacing: although most of s type is poisoning. For d type most is eating.
* The distribution of our numerical columns is right-skewed with outliers, meaning that most values are concentrated on the left side of the distribution, but there are some unusually high values (outliers) that are far away from the rest. This suggests that our data may not be normally distributed, which could impact our analysis and modeling results.
* There are most mushrooms in spring and autumn.
* Only in winter more mushrooms is eating than poisoning.

III DATA PRE-PROCESSING
Conclusions:
* Standardizing the Missing Data with null values to make it easier to handle.
* Removing duplicates and unnecessary columns.
* Aggregating categorical and numerical columns
* To maximize the accuracy of our predictions, we will replace all missing values in categorical columns with 'Unknown', allowing us to retain as many columns as possible for analysis.
* Convert types (downcasting).

IV DATA PROCESSING
Conclusions:
* Principal Component Analysis or PCA is a linear dimensionality reduction algorithm. In this technique, data is linearly transformed onto a new coordinate system such that the directions (principal components) capturing the largest variation in the data can be easily identified.
* We can preserve 99% of variance with 50 components if we use PCA.
* There is a lot of feature variables so instead of engineering new features, we might want to focus on eliminating uninformative features and focusing on only the crucial ones. PCAs might also be useful in this scenario where it allows us to pick out components which convey useful information about the data.
* Some of the features show close to 0 correlation with the target variable which could signal that it is useless to us.
* Mutual Information Score helps us to understand how much the feature variables tell us about the target variable.
Since our data have a lot of feature variables, we can take help of this to remove redundant feature variables. This may improve the proformance of our model.



The goal is to predict whether a mushroom is edible or poisonous based on its physical features, such as color, shape, and texture.
To tackle this problem, we'll be analyzing a special dataset. This dataset was created by a deep learning model that studied thousands of mushrooms. While the data is similar to a well-known mushroom dataset, there are some differences that make this project unique.
This project will be implemented based on a real dataset, provided by the project organizer. The goal is to develop a model that can classify mushrooms as edible ('e')=0 or poisonous ('p')=1 using a set of physical attributes provided in the dataset.

In  was  develop the pipeline based on  classification models.
Models was split into 5 parts: 
<ul>
V.1 XGBClassifier:
<li> Best score is for XGBClassifier Tuning - Optuna:
    <li> Train Score = 99.0175%
    <li> Valid Score = 98.82%
    <li> Mean Squared Error = 0.0118
    <li> Matthews correlation coefficient <mark style="background-color:#A1D6E2;color:white;border-radius:5px;opacity:1.0">(MCC) = 0.974</mark>
    <li> ROC AUC = 0.9956
    <li> FN = 44
    <li> FP = 74
</ul>
<ul>
V.2 CatBoostClassifier:
<li> Train Score = 91.1075%
<li> Valid Score = 90.44%
<li> Mean Squared Error = 0.0956
<li> Matthews correlation coefficient <mark style="background-color:#A1D6E2;color:white;border-radius:5px;opacity:1.0">(MCC) = 0.807</mark>
<li> ROC AUC = 0.9654
 <li> FN = 405
<li> FP = 551
</ul>
<ul>
V.3 RandomForestClassifier Tuning - Optuna:
<li> Train Score = 99.155%
<li> Valid Score = 98.67%
<li> Mean Squared Error = 0.0133
<li> Matthews correlation coefficient <mark style="background-color:#A1D6E2;color:white;border-radius:5px;opacity:1.0">(MCC) = 0.972</mark>
<li> ROC AUC = 0.9941
<li> FN = 53
<li> FP = 80
</ul>
<ul>
V.4 LGBMClassifier uning - Optuna:
<li> Train Score = 99.9475%
<li> Valid Score = 98.81%
<li> Mean Squared Error = 0.0119
<li> Matthews correlation coefficient <mark style="background-color:#A1D6E2;color:white;border-radius:5px;opacity:1.0">(MCC) = 0.976</mark>
<li> ROC AUC = 0.9947
<li> FN = 49
<li> FP = 70
</ul>
<ul>
V.5 VotingClassifier 2:
<li> Train Score = 99.645%
<li> Valid Score = 99.19%
<li> Mean Squared Error = 0.0081
<li> Matthews correlation coefficient <mark style="background-color:red;color:white;border-radius:5px;opacity:1.0">(MCC) = 0.9837</mark>
<li> Accuracy = 0.991
<li> FN = 53
<li> FP = 38
</ul>

We started with classifiers for which we obtained an MCC score  80% for Catboost. After that it was changed preprocessor and the results improved significantly.
We followed with an series of traditional classification models for which scores was aound 97%.

At the end was tested models based on VotingClassifier.
This model is based on the three best ones: XGBClassifier, RandomForestClassifier and  LGBMClassifier.  As a result we obtained an MCC score of 98.4% which is the  best result from all tests.


The score AUC is defenitely the best for threshold = 0.5, and if we look at the confusion matrix we can see false negative score is very small which is magnificent for us.   
    
In case where the most important thing will be to catch as many poisonus mushrooms as it is possible without looking at the costs of such an operation. In such a special case we should check as many FN as we can, but at the same time we want to have as few FP values as it is possible. Because of that MCC or even AUC is not crucial metric in this case. Crucial is to minimize prediction for FP, we don't want to loose poisonus mushrooms which will be predicted as not positive. The best metric in this case will be this with threshold = 0.2. Unfortunatelly FN indicator has more values than for threshold = 0.5,  and our MCC and AUC will fall a bit but like I said earlier in such special case it is not relevant.
</div>


### Author Details:
- Name: Adam Smulczyk
- Email: adam.smulczyk@gmail.com
- Profile: [Github](https://github.com/AdamSmulczyk)
- [Github Repository](https://github.com/AdamSmulczyk/014_Poisonous_Mushrooms)