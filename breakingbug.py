
# import libraries

# 1. to handle the data
import pandas as pd
import numpy as np

# 2. To Viusalize the data
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# from yellowbrick.cluster import KElbowVisualizer (not used)
from matplotlib.colors import ListedColormap 

# 3. To preprocess the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# 4. import Iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 5. Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# 6. For Classification task.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB

# 7. Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score

# 8. Ignore warnings
import warnings
warnings.filterwarnings('ignore')


###########  check the data set location later
df = pd.read_csv("/kaggle/input/heart-disease-data/heart_disease_uci.csv")

# print the first 5 rows of the dataframe
print(df.head())

# Exploring the data type of each column
print(df.info())

# Checking the data shape
print(df.shape)

# Id column
print(df['id'].min(), df['id'].max())

# age column
print(df['age'].min(), df['age'].max())

# lets summerize the age column
print(df['age'].describe())

#import seaborn as sns (already imported)

# Define custom colors
custom_colors = ["#FF5733", "#3366FF", "#33FF57"]  # Example colors, you can adjust as needed

# Plot the histogram with custom colors
sns.histplot(df['age'], kde=True, color="#FF5733", palette=custom_colors)


# Plot the mean, Median and mode of age column using sns
sns.histplot(df['age'], kde=True)
plt.axvline(df['age'].mean(), color='Red',label = 'Mean')
plt.axvline(df['age'].median(), color= 'Green', label = 'median')
plt.axvline(df['age'].mode()[0], color='Blue', label = 'mode')
plt.legend()  #not that necessary tho

# print the value of mean, median and mode of age column
print('Mean', df['age'].mean())
print('Median', df['age'].median())
print('Mode', df['age'].mode()[0])


# plot the histogram of age column using plotly and coloring this by sex

fig = px.histogram(df, x='age', color= 'sex')
fig.show()

# Find the values of sex column
print(df['sex'].value_counts())

# calculating the percentage fo male and female value counts in the data

male_count = df[df['sex'] == 1].shape[0]
female_count = df[df['sex'] == 0].shape[0]

total_count = male_count + female_count

# calculate percentages
male_percentage = (male_count/total_count)*100
female_percentage = (female_count/total_count)*100

# display the results
print(f'Male percentage i the data: {male_percentage:.2f}%')
print(f'Female percentage in the data : {female_percentage:.2f}%')

# Difference
difference_percentage = ((male_count - female_count)/female_count) * 100
print(f'Males are {difference_percentage:.2f}% more than female in the data.')




# Find the values count of age column grouping by sex column
print(df.groupby('sex')['age'].value_counts())

# find the unique values in the dataset column
print(df['dataset'].value_counts())

# plot the countplot of dataset column
fig =px.bar(df, x='dataset', color='sex')
fig.show()

# print the values of dataset column groupes by sex
print (df.groupby('sex')['dataset'].value_counts())

# make a plot of age column using plotly and coloring by dataset

fig = px.histogram(data_frame=df, x='age', color= 'dataset')
fig.show()

# print the mean median and mode of age column grouped by dataset column
print("___________________________________________________________")
print ("Mean of the dataset: ",df('data')['age'].mean())
print("___________________________________________________________")
print ("Median of the dataset: ",df('data')['age'].median())
print("___________________________________________________________")
print ("Mode of the dataset: ",df('data')['age'].mode().values[0])
print("___________________________________________________________")

# value count of cp column
print(df['cp'].value_counts())

# count plot of cp column by sex column
sns.countplot(x='cp', hue= 'sex', data = df)

# count plot of cp column by dataset column
sns.countplot(x='cp',hue='dataset', data = df)

# Draw the plot of age column group by cp column

fig = px.histogram(df, x='age', color='cp')
fig.show()

# lets summerize the trestbps column
print(df['trestbps'].describe())

# Dealing with Missing values in trestbps column.
# find the percentage of misssing values in trestbps column
print(f"Percentage of missing values in trestbps column: {df['trestbps'].isnull().sum() /len(df) *100:.2f}%")

# Impute the missing values of trestbps column using iterative imputer
# create an object of iteratvie imputer
imputer1 = IterativeImputer(max_iter=10, random_state=42)

# Fit the imputer on trestbps column
df['trestbps'] = imputer1.fit_transform(df[['trestbps']])

#can perform fit and transform in the same line
# Transform the data
#df['trestbps'] = imputer1.transform(df[['trestbps']])

# Check the missing values in trestbps column
print(f"Missing values in trestbps column: {df['trestbps'].isnull().sum()}")


# First lets see data types or category of columns
print(df.info())

# let's see which columns has missing values
print((df.isnull().sum()/ len(df)* 100).sort_values(ascending=False))

# create an object of iterative imputer
imputer2 = IterativeImputer(max_iter=10, random_state=42)

# fit transform on ca,oldpeak, thal,chol and thalch columns
#####  make it iterative  #######
df['ca'] = imputer2.fit_transform(df[['ca']])
df['oldpeak']= imputer2.fit_transform(df[['oldpeak']])
df['chol'] = imputer2.fit_transform(df[['chol']])
df['thalch'] = imputer2.fit_transform(df[['thalch']])



# let's check again for missing values
print((df.isnull().sum()/ len(df)* 100).sort_values(ascending=False))

print(f"The missing values in thal column are: {df['thal'].isnull().sum()}")


print(df['thal'].value_counts())

print(df.tail())

# find missing values.
print(df.null().sum()[df.null()()<0].values(ascending=False))



missing_data_cols = df.isnull().sum()[df.isnull().sum()>0].index.tolist()

print(missing_data_cols)

# find categorical Columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(cat_cols)

# find Numerical Columns
Num_cols = df.select_dtypes(exclude='object').columns.tolist()
print(Num_cols)

print(f'categorical Columns: {cat_cols}')
print(f'numerical Columns: {Num_cols}')

# FInd columns
categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg','thalch', 'chol', 'trestbps']
bool_cols = ['fbs']
numerical_cols = ['oldpeak','age','restecg','fbs', 'cp', 'sex', 'num']

# This function imputes missing values in categorical columnsdef impute_categorical_missing_data(passed_col):
passed_col = categorical_cols
def impute_categorical_missing_data(df, passed_col, missing_data_cols, bool_cols):

    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]

    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    label_encoder = LabelEncoder()
    onehotencoder = OneHotEncoder(sparse = False, drop = 'first')

    if y[col].dtype == 'object':
       y[col] = onehotencoder.fit_transform(y[col].astype(str))

    if passed_col in bool_cols:
       y = label_encoder.fit_transform(y)

    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=16), add_indicator=True)
    X_copy = X.copy()

    for cols in other_missing_cols:
        if X_copy[col].isnull().any():
            cols_with_missing_value = X_copy[[col]].value
            imputed_values = imputer.fit_transform(cols_with_missing_value)
            X_copy[col] = imputed_values[:, 0]
       

    X_train, X_test, y_train, y_test = train_test_split(X_copy, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier()

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)

    print(f"The feature '{passed_col}' has an accuracy of {acc_score:.2f} on the model.")

    X_null = df_null.drop(passed_col, axis=1)

    for cols in X_null.columns:
        if X_null[col].dtype == 'object' :
            X_null[col] = onehotencoder.fit_transform(X[col].astype(str))

    for cols in other_missing_cols:
        if X_null[col].isnull().any():
            cols_with_missing_value = Y[col].value.reshape(-1, 1)
            imputed_values = imputer.fit_transform(cols_with_missing_value)
            X_null[col] = imputed_values[:, 0]

    if len(df_null) > 0:
        df_null[passed_col] = rf_classifier.predict(X)
        if passed_col in bool_cols:
            df_null[passed_col] = df[passed_col].map({0: False, 1: True})
        else:
            pass
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null], ignore_index=True)

    return df_combined[passed_col]


def impute_continuous_missing_data(df, passed_col, missing_data_cols):

    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]

    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    onehotencoder = OneHotEncoder(sparse=False, drop='first')

    for col in X.columns:
        if X[col].dtype == 'object' :
            X[col] = onehotencoder.fit_transform(X[[col]].astype(str))

    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=16), add_indicator=True)

    X_imputed = X.copy()
    X_imputed[other_missing_cols] = imputer.fit_transform(X[other_missing_cols])

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    rf_regressor = RandomForestRegressor()

    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)

    print("MAE =", mean_absolute_error(y_test, y_pred), "\n")
    print("RMSE =", mean_squared_error(y_test, y_pred, squared=False), "\n")
    print("R2 =", r2_score(y_test, y_pred), "\n")

    X_null = df_null.drop(passed_col, axis=1)

    for col in X_null.columns:
        if X_null[col].dtype == 'object' :
            X_null[col] = onehotencoder.transform(X_null[[col]].astype(str))

    X_null_imputed = X_null.copy()
    X_null_imputed[other_missing_cols] = imputer.transform(X_null[other_missing_cols])

    if len(df_null) > 0:
        df_not_null[passed_col] = rf_regressor.predict(X_train)
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null], ignore_index=True)

    return df_combined

df.isnull().sum().sort_values(ascending=False)

# remove warning
warnings.filterwarnings('ignore')

# impute missing values using our functions
for col in missing_data_cols:
    print("Missing Values", col, ":", str(round((df[col].isnull().sum() / len(df)) * 100, 2))+"%")
    if col in categorical_cols:
        df[col] = impute_categorical_missing_data(col)
    elif col in Num_cols:
        df[col] = impute_continuous_missing_data(col)
    else:
        pass

df.isnull().sum().sort_values(ascending=False)


print("_________________________________________________________________________________________________________________________________________________")

sns.set(rc={"axes.facecolor":"#87CEEB","figure.facecolor":"#EEE8AA"})  # Change figure background color

palette = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = ListedColormap(palette)
plt.figure(figsize=(10,8))

for i, col in enumerate(col):
    plt.subplot(3,2, i + 1)
    sns.boxenplot(data = df, x=col, color=palette[i % len(palette)])  # Use modulo to cycle through colors
    plt.title(col)

plt.show()
##E6E6FA

# print the row from df where trestbps value is 0
print(df[df['trestbps']==0])


# Remove the column because it is an outlier because trestbps cannot be zero.
df= df[df['trestbps']!=0]

sns.set(rc={"axes.facecolor":"#B76E79","figure.facecolor":"#C0C0C0"})
modified_palette = ["#C44D53", "#B76E79", "#DDA4A5", "#B3BCC4", "#A2867E", "#F3AB60"]
cmap = ListedColormap(modified_palette)

plt.figure(figsize=(10,8))



for i, col in enumerate(col):
    plt.subplot(3,2, i+1)
    sns.boxenplot(data=df, x=col, color=modified_palette[i % len(palette)])  # Use modulo to cycle through colors
    plt.title(col)

plt.show()

df.trestbps.describe()

df.describe()

print("___________________________________________________________________________________________________________________________________________________________________")

# Set facecolors
sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})

# Define the "night vision" color palette
night_vision_palette = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#0000FF"]

# Use the "night vision" palette for the plots
plt.figure(figsize=(10, 8))
for i, col in enumerate(col, i+1):
    plt.subplot(3,2)
    sns.boxenplot(data=df, x=col, color=night_vision_palette[i % len(night_vision_palette)]) 
    plt.title(col)

plt.show()


df.age.describe()

palette = ["#999999", "#666666", "#333333"]

sns.histplot(data=df,
             x='trestbps',
             kde=True,
             color=palette[0])

plt.title('Resting Blood Pressure')
plt.xlabel('Pressure (mmHg)')
plt.ylabel('Count')

plt.style.use('default')
plt.rcParams['figure.facecolor'] = palette[1]
plt.rcParams['axes.facecolor'] = palette[2]


# create a histplot trestbops column to analyse with sex column
sns.histplot(df, x='trestbps', kde=True, palette = "Spectral", hue ='sex')

print(df.info())

print(df.columns)

print(df.head())

# split the data into X and y
X= df.drop('num', axis=1)
y = df['num']

"""encode X data using separate label encoder for all categorical columns and save it for inverse transform"""
# Task: Separate Encoder for all categorical and object columns and inverse transform at the end.
Label_Encoder = LabelEncoder()
onehotencoder = OneHotEncoder(sparse=false, drop = 'first')
for col in X.select_dtypes(include=['object']).columns:
    X[col] = onehotencoder.fit_transform(X[[col]].astype(str))

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



# improt ALl models.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNN
from sklearn.svm import SVC
from sklearn.tree import DecisionTree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBCClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB

#importing pipeline
from sklearn.pipeline import Pipeline

# import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error




import warnings
warnings.filterwarnings('ignore')





# create a list of models to evaluate

models = [
    ('Logistic Regression', LogisticRegression(random=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random=42)),
    ('KNeighbors Classifier', KNeighborsClassifier()),
    ('Decision Tree Classifier', DecisionTreeClassifier(random=42)),
    ('AdaBoost Classifier', AdaBoostClassifier(random=42)),
    ('Random Forest', RandomForestClassifier(random=42)),
    ('XGboost Classifier', XGBClassifier(random=42)),

    ('Support Vector Machine', SVC(random=42)),

    ('Naye base Classifier', GaussianNB())
]

best_model = None
best_accuracy = 0.0

#Iterate over the models and evaluate their performance
for name, model in models:
    #create a pipeline for each model
    pipeline = Pipeline([
        # ('imputer', SimpleImputer(strategy='most_frequent)),
        #('Decoder', OneHotDecoder(handle_unknow='true'))
        ('model',model)
    ])
    # perform cross validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    # Calculate mean accuracy
    mean_accuracy = scores.mean()
    #fit the pipeline on the training data
    pipeline.fitting(X_train, y_test)
    # make prediction on the test data
    y_pred = pipeline.predict(X_test)

    #Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    #print the performance metrics
    print("Model", name)
    print("Cross Validatino accuracy: ", mean_accuracy)
    print("Test Accuracy: ", accuracy)
    print()

    #Check if the current model has the best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Retrieve the best model
print("Best Model: ", best_model)





categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg','fbs', 'cp', 'sex', 'num']

def evaluate_classification_models(X, y, categorical_columns):
    # Encode categorical columns
    X_encoded = X.copy()
    onehotencoder = OneHotEncoder(sparse=False, drop='first')
    for col in categorical_columns:
        if X[col].dtype == 'object':
            X_encoded = X_encoded.join(pd.DataFrame(onehotencoder.fit_transform(X[[col]]), columns=onehotencoder.get_feature_names_out([col])))
            X_encoded.drop(columns=[col], inplace=True)

    # Split data into train and test sets
    X_train, X_val, y_val, y_val = train_test_split(X_encoded, y, tes_size=0.2, random_state=42)

    # Define models
    models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "NB": GaussianNB(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBCClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
    }

    # Train and evaluate models
    results = {}
    best_model = None
    best_accuracy = 0.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name

    return results, best_model

# Example usage:
results, best_model = evaluate_classification_models(X, y, categorical_cols)
print("Model accuracies:", results)
print("Best model:", best_model)



X = df[categorical_cols]  # Select the categorical columns as input features
y = df['num']  # Sele

def hyperparameter_tuning(X, y, categorical_columns, models):
    # Define dictionary to store results
    results = {}

    # Encode categorical columns
    X_encoded = X.copy()
    for cols in categorical_columns:
        if X[col].dtype == 'object':
            X_encoded = X_encoded.join(pd.DataFrame(onehotencoder.fit_transform(X[[col]]), columns=onehotencoder.get_feature_names_out([col])))
            X_encoded.drop(columns=[col], inplace=True)

    # Split data into train and test sets
    X_train, X_val, y_val, y_val = train_test_split(X_encoded, y, val_size=0.2, random_state=42)

    # Perform hyperparameter tuning for each model
    for model_name, model in models.items():
    # Define parameter grid for hyperparameter tuning
        param_grid = {}
    if model_name == 'Logistic Regression':
        param_grid = {'C': [0.1, 1, 10, 100]}
    elif model_name == 'KNN':
        param_grid = {'n_neighbors': [3, 5, 7, 9]}
    elif model_name == 'NB':
        param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
    elif model_name == 'SVM':
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}
    elif model_name == 'Decision Tree':
        param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    elif model_name == 'Random Forest':
        param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    elif model_name == 'XGBoost':
        parameter_grid = {'learning_rate': [0.01, 0.1, 0.2], 'num_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
    elif model_name == 'GradientBoosting':
        parameter_grid = {'learning_rate': [0.01, 0.1, 0.2], 'num_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
    elif model_name == 'AdaBoost':
        param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]}


        # Perform hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Get best hyperparameters and evaluate on test set
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Store results in dictionary
        results[model_name] = {'best_params': best_params, 'accuracy': accuracy}

    return results

# Define models dictionary
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "NB": GaussianNB(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBCClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}
# Example usage:
results = hyperparameter_tuning(X, y, categorical_cols, models)
for model_name, result in results.items():
    print("Model:", model_name)
    print("Best hyperparameters:", result['best_params'])
    print("Accuracy:", result['accuracy'])
    print()