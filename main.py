import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
print("Current Working Directory:", os.getcwd())

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(df.head())
print(df.info())
print(df.isnull().sum())

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
df.reset_index(drop=True , inplace=True)

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'gender']
for col in binary_cols:
    if df[col].nunique() ==2:
        df[col]=df[col].map({'Yes':1, 'No':0})
    elif col == 'gender':
        df[col]=df[col].map({'Female':1, 'Male':0})

df = pd.get_dummies(df, columns=['MultipleLines', 'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                                 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract','PaymentMethod'], drop_first=True)


from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_cols] = scalar.fit_transform(df[num_cols])

print(df.head())
print(df.shape)

sns.countplot(data=df, x='Churn')
plt.title('Distribution Of Churns')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

Churn_percentage = df['Churn'].value_counts(normalize=True)*100
print(Churn_percentage)

#analyzing individual features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], bins=30 , kde=True)
    plt.title('Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

#catogerial features

categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                        'PhoneService', 'MultipleLines', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 
                        'Contract', 'PaperlessBilling', 'PaymentMethod']

for features in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=feature, hue='Churn')
    plt.title(f'{feature} vs Churn')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Churn')
    plt.tight_layout()
    plt.show()

for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x='Churn', y=feature)
    plt.title(f'{feature} by Churn')
    plt.xlabel('Churn')
    plt.ylabel(feature)
    plt.show()

#relationship between numerical features

corr_matrix = df[numerical_features].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#model building for prediction
x = df.drop('Churn', axis=1)
y = df['Churn']

from sklearn.model_selection import train_test_split
x_train , x_test , y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train,y_train)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1, random_state=42)
gb_model.fit(x_train,y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Example: Evaluating Logistic Regression
y_pred_log = log_model.predict(x_test)
y_prob_log = log_model.predict_proba(x_test)[:, 1]

print("Logistic Regression Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("F1 Score:", f1_score(y_test, y_pred_log))
print("ROC AUC:", roc_auc_score(y_test, y_prob_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))


