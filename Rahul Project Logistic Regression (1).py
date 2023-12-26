#!/usr/bin/env python
# coding: utf-8

# ## The objective of this project is to develop a logistic regression model to predict whether a particular internet user is likely to click on an advertisement based on their demographic and online behavior features. 
# 
# The dataset provided includes the following features:
# 
# Daily Time Spent on Site: The amount of time (in minutes) a consumer spends on the site.
# Age: The age of the customer in years.
# Area Income: The average income of the geographical area of the consumer.
# Daily Internet Usage: The average number of minutes a consumer spends on the internet per day.
# Ad Topic Line: The headline of the advertisement.
# City: The city of the consumer.
# Male: A binary indicator for whether the consumer is male (1 for male, 0 for female).
# Country: The country of the consumer.
# Timestamp: The time at which the consumer either clicked on the ad or closed the window.
# Clicked on Ad: The target variable, where 1 indicates the consumer clicked on the ad, and 0 indicates they did not.
# Using this dataset, the goal is to build a logistic regression model that accurately predicts whether a given user will click on an ad based on the provided features. This model will be valuable for optimizing ad targeting and improving the effectiveness of advertising campaigns.
# 
# The steps involved in achieving this goal include data preprocessing, exploratory data analysis, feature selection/engineering, model training and evaluation, and finally, deploying the model for practical use in ad campaigns.
# 
# Ultimately, the success of this project will be measured by the model's ability to accurately classify users into the 'Clicked on Ad' categories, as well as its potential to enhance the efficiency of online advertising efforts.

# ## Importing the important Libraries and Packages 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

#Importing the dataset Advertisement containing the data to be analyzed 
data = pd.read_csv("C:/Users/Rahul Nain/OneDrive/Desktop/Personal/Data Scinece Related Info/PROJECTS FOR CV/Logistic Regression Ad Prediction/advertising.csv")


# Having a prelimnary analysis of the given dataset to understand it structure and composition 

# In[3]:


data.info()


# Basic Description of the data 

# In[4]:


data.describe()


# ## Exploratory Data Analysis to study the basic information about the dataset and understanding its compositions any correlations or visible trends existing and how it can shape our studies. 
# 
# EDA is a crucial initial step in understanding our dataset. It helps us grasp its size, attributes, and composition.
# By identifying correlations and patterns, we gain insights that guide feature selection and model design. 
# EDA also uncovers outliers and informs data cleaning.
# 
# Key Objectives:
# 
# Understand dataset composition and feature types.
# Uncover correlations and influential features.
# Inform decision-making for subsequent stages.
# Strategies:
# 
# Use descriptive statistics for initial data overview.
# Employ visualizations (e.g., graphs, scatter plots) to inspect relationships.
# Analyze categorical features through frequency distributions.
# Perform correlation analysis for numerical attributes.
# Impact on Project:
# 
# Guides feature selection and engineering.
# Aids in model selection and parameter optimization.
# Identifies outliers for special handling.
# Provides powerful visualizations for stakeholder presentation.
# In essence, EDA is our project compass, ensuring informed decisions and alignment with the goal of building an 
# effective ad click prediction model.

# In[12]:


plt.figure(figsize=(8, 6))
data.Age.hist(bins=data.Age.nunique(), color='purple')
plt.xlabel('Age', color='black')
plt.ylabel('Frequency', color='black')
plt.show()

# Adding a line curve
data.Age.plot(kind='kde', color='blue', linewidth=2)  # Kernel Density Estimate (KDE) plot

plt.show()


# The frequency distribution curve for age is roughly symmetrical around the mean of 36. This indicates that most individuals in the dataset cluster around the average age. The distribution is relatively even on both sides of the mean, suggesting a balanced spread of ages.

# In[13]:


plt.figure(figsize=(8, 6))
sns.jointplot(x=data["Area Income"], y=data.Age)


# The joint plot reveals a concentration of data points in the age range of 20 to 40 and income levels between 40,000 to 80,000. This suggests that a significant portion of the sample comprises younger individuals with moderate to middle-range incomes.

# In[14]:


plt.figure(figsize=(8, 6))
sns.jointplot(x=data["Daily Time Spent on Site"], y=data.Age, kind='kde')


# The joint plot indicates that the highest amount of time spent on the site is primarily by individuals in the age group of 20 to 40. This suggests that younger users tend to spend more time on the site compared to other age groups.

# In[15]:


plt.figure(figsize=(8, 6))
sns.jointplot(x=data["Daily Time Spent on Site"], y=data["Daily Internet Usage"])


# The joint plot reveals two distinct clusters in the dataset. One group spends approximately 100 to 175 minutes on the internet and around 30 to 70 minutes on the site. The second group invests more time, roughly 175 to 275 minutes, on the internet and spends 65 to 90 minutes on the site. This division indicates a clear pattern in user behavior regarding internet usage and time spent on the site.

# In[16]:


sns.pairplot(data, hue='Clicked on Ad')


# In[17]:


data['Clicked on Ad'].value_counts()


# Category 0: There are 500 instances where the outcome is 0. This could represent a category like "Did Not Click on Ad".
# 
# Category 1: There are 500 instances where the outcome is 1. This could represent a category like "Clicked on Ad".
# 
# A balanced distribution (where both categories have roughly equal counts) is generally desirable, as it helps the model learn from both classes equally. If there's a severe imbalance, the model might be biased towards the more prevalent class.

# In[18]:


plt.figure(figsize=(10, 7))
sns.heatmap(data.corr(), annot=True)


# The correlation matrix highlights strong positive relationships. Daily time spent on site and ads clicked exhibit a notable correlation of 0.75, indicating a significant connection. Similarly, daily internet usage and ads clicked display a substantial correlation of 0.79, signifying a robust association between these variables.

# # Logistic Regression:
# Logistic Regression is a statistical technique used for predicting the probability of a binary outcome (1/0, Yes/No, True/False) based on one or more predictor variables. It's widely employed in classification tasks where the goal is to assign an instance to one of two possible categories.
# 
# # Use and Application:
# In our project, Logistic Regression can help predict whether a user will click on an advertisement based on features like age, time spent, income, etc. It models the probability of a user clicking on an ad given the feature values.

# # Overall Flow:
# # Understanding the Business Problem:
# 
# Define the problem (predicting ad clicks) and its significance for the business.
# Data Collection and Exploration: Obtain the dataset and perform exploratory data analysis (EDA) to understand the data's characteristics.
# Preprocessing and Feature Engineering: Handle missing data, encode categorical variables, and engineer features based on EDA findings.
# Data Splitting: Divide the data into training and testing sets to train and evaluate the model.
# Model Building: Apply Logistic Regression to the training data.
# Model Evaluation: Assess the model's performance on the testing set using appropriate metrics.
# Deployment: Deploy the final model for practical use in ad campaigns.
# Monitoring and Maintenance: Continuously monitor the model's performance and update as needed to ensure accurate predictions.

# In[19]:


x = np.linspace(-6, 6, num=1000)
plt.figure(figsize=(10, 6))
plt.plot(x, (1 / (1 + np.exp(-x))))
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sigmoid Function")


# # Logistic Regression Theory
# Logistic Regression is a widely used statistical technique for binary classification problems. Unlike linear regression, which predicts continuous values, logistic regression predicts the probability of an instance belonging to a specific class. It achieves this through the application of a sigmoid function, which compresses the output between 0 and 1, effectively representing probabilities.
# 
# # Sigmoid Function
# The sigmoid function takes a linear combination of features and their associated weights. This weighted sum, combined with a bias term, is fed into the sigmoid function. The result of this transformation is a probability that the instance belongs to the positive class. The decision boundary is set at 0.5, meaning that if the predicted probability is greater than or equal to 0.5, the instance is classified as belonging to the positive class; otherwise, it is classified as belonging to the negative class.
# 
# The goal of logistic regression is to find the optimal weights that minimize the error in predicting the target variable. This is typically achieved through techniques like Maximum Likelihood Estimation (MLE) or gradient descent. The cost function used in logistic regression is derived from the likelihood function and is designed to penalize predictions that deviate significantly from the actual labels.
# 
# Logistic regression offers several advantages, including the interpretability of coefficients, efficiency for linearly separable data, and low computational cost. However, it comes with limitations, such as the assumption of linearity, its primary suitability for binary classification tasks, and sensitivity to outliers. In summary, logistic regression is a powerful tool for binary classification problems, leveraging the sigmoid function to model probabilities and make predictions.

# Splitting the data into Train and Test and preparing for logistic regression. 

# In[28]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# In[21]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split


X = data.drop(['Timestamp', 'Clicked on Ad', 'Ad Topic Line', 'Country', 'City'], axis=1)
y = data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# cat_columns = []
num_columns = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']


ct = make_column_transformer(
    (MinMaxScaler(), num_columns),
    (StandardScaler(), num_columns),
    remainder='passthrough'
)

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)


# In[22]:


from sklearn.linear_model import LogisticRegression


lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)

print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)


# The logistic regression model demonstrates exceptional performance on both the training and test sets. On the training set, the model achieves an impressive accuracy of 97.43%. This indicates that nearly 97.43% of instances in the training data are correctly classified. Additionally, the precision, recall, and F1 score, which are key metrics for binary classification, are all very high, with values of 0.9745, 0.9742, and 0.9742 respectively. These metrics further confirm the model's ability to accurately identify instances belonging to the positive class.
# 
# Moving on to the test set, the model maintains a high level of accuracy at 97.00%. This signifies that the model generalizes well to new, unseen data. The precision, recall, and F1 score on the test set are also impressive, with values of 0.970204, 0.970000, and 0.970005 respectively. These metrics indicate a strong performance in correctly identifying instances in the positive class while minimizing false positives.
# 
# Overall, the logistic regression model exhibits robust predictive capabilities, as demonstrated by its high accuracy and consistently high precision, recall, and F1 scores on both the training and test sets. This suggests that the model is well-suited for accurately predicting whether a user will click on an advertisement based on the provided features.

# In[23]:


from sklearn.ensemble import RandomForestClassifier


rf_clf = RandomForestClassifier(n_estimators=1000)
rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)


# # 5. Performance Measurement
# 
# #### 1. Confusion Matrix
# - Each row: actual class
# - Each column: predicted class
# 
# First row: Non-clicked Ads, the negative class:
# * 143 were correctly classified as Non-clicked Ads. **True negatives**. 
# * Remaining 6 were wrongly classified as clicked Ads. **False positive**
# 
# 
# Second row: The clicked Ads, the positive class:
# * 3 were incorrectly classified as Non-clicked Ads. **False negatives**
# * 146 were correctly classified clicked Ads. **True positives**
# 
# #### 2. Precision
# 
# **Precision** measures the accuracy of positive predictions. Also called the `precision` of the classifier ==> `98.01%`
# 
# $$\textrm{precision} = \frac{\textrm{True Positives}}{\textrm{True Positives} + \textrm{False Positives}}$$
# 
# #### 3. Recall
# 
# `Precision` is typically used with `recall` (`Sensitivity` or `True Positive Rate`). The ratio of positive instances that are correctly detected by the classifier.
# 
# $$\textrm{recall} = \frac{\textrm{True Positives}}{\textrm{True Positives} + \textrm{False Negatives}}$$ ==> `96.10%`
# 
# #### 4. F1 Score
# 
# $F_1$ score is the harmonic mean of precision and recall. Regular mean gives equal weight to all values. Harmonic mean gives more weight to low values.
# 
# 
# $$F_1=\frac{2}{\frac{1}{\textrm{precision}}+\frac{1}{\textrm{recall}}}=2\times \frac{\textrm{precision}\times \textrm{recall}}{\textrm{precision}+ \textrm{recall}}=\frac{TP}{TP+\frac{FN+FP}{2}}$$ ==> `97.05%`
# 
# The $F_1$ score favours classifiers that have similar precision and recall.
# 
# #### 5. Precision / Recall Tradeoff
# 
# Increasing precision reduced recall and vice versa

# In[24]:


from sklearn.metrics import precision_recall_curve


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.title("Precisions/recalls tradeoff")

    
precisions, recalls, thresholds = precision_recall_curve(y_test, lr_clf.predict(X_test))


plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.subplot(2, 2, 2)
plt.plot(precisions, recalls)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("PR Curve: precisions/recalls tradeoff");


# Precision-Recall vs. Threshold:
# 
# This plot shows how precision and recall change as the threshold for classifying a sample as positive or negative changes.
# You can observe how adjusting the threshold affects the tradeoff between precision and recall.
# Precision vs. Recall Curve:
# 
# This is a classic Precision-Recall Curve. It shows how precision and recall are related to each other.
# Ideally, you'd want a high precision and high recall, but often there's a tradeoff (i.e., improving one may come at the expense of the other).

# In[25]:


from sklearn.metrics import roc_curve


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

fpr, tpr, thresholds = roc_curve(y_test, lr_clf.predict(X_test))
plt.figure(figsize=(9,6)); 
plot_roc_curve(fpr, tpr)
plt.show();


# The Receiver Operating Characteristics (ROC) Curve
# Instead of plotting precision versus recall, the ROC curve plots the true positive rate (another name for recall) against the false positive rate. The false positive rate (FPR) is the ratio of negative instances that are incorrectly classified as positive. It is equal to one minus the true negative rate, which is the ratio of negative instances that are correctly classified as negative.
# 
# The TNR is also called specificity. Hence the ROC curve plots sensitivity (recall) versus 1 - specificity.

# In[26]:


from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, lr_clf.predict(X_test))


# In[27]:


from sklearn.model_selection import GridSearchCV


lr_clf = LogisticRegression()

penalty = ['l1', 'l2']
C = [0.5, 0.6, 0.7, 0.8]
class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
solver = ['liblinear', 'saga']

param_grid = dict(
    penalty=penalty, 
    C=C, 
    class_weight=class_weight, 
    solver=solver
)

lr_cv = GridSearchCV(
    estimator=lr_clf, 
    param_grid=param_grid, 
    scoring='f1',
    verbose=1, 
    n_jobs=-1, 
    cv=10
)

lr_cv.fit(X_train, y_train)
best_params = lr_cv.best_params_
print(f"Best parameters: {best_params}")

lr_clf = LogisticRegression(**best_params)
lr_clf.fit(X_train, y_train)

print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)

