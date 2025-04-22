# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![Screenshot 2025-04-22 111353](https://github.com/user-attachments/assets/240a7fee-ceb6-43d6-b1e5-b76be62a3ada)

```
df.dropna()
```
![Screenshot 2025-04-22 111440](https://github.com/user-attachments/assets/4da01cbb-bc5a-44d2-b78d-6103555a3139)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![Screenshot 2025-04-22 111510](https://github.com/user-attachments/assets/34f0e8a6-3f5b-4ef9-9cd3-8a8e7816f110)

```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df=pd.read_csv("/content/bmi (1).csv")
df[['Height', 'Weight']] = sc.fit_transform(df[['Height', 'Weight']])
print(df.head(10))
```
![Screenshot 2025-04-22 111544](https://github.com/user-attachments/assets/2f8c028a-abf7-4c82-bc80-c3317b3cf966)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2025-04-22 111611](https://github.com/user-attachments/assets/491281b6-fcd1-4c03-82d2-e83dd1cf2a26)

```
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![Screenshot 2025-04-22 111751](https://github.com/user-attachments/assets/a2bd164b-7d1c-4f38-9e2c-709f34e67813)

```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![Screenshot 2025-04-22 111838](https://github.com/user-attachments/assets/c43ce6c3-acb6-40eb-a7da-b5686d7b4bb1)

```
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![Screenshot 2025-04-22 111907](https://github.com/user-attachments/assets/66208af3-e557-4537-803c-5457d8dfa93e)

```
contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)
```
![Screenshot 2025-04-22 112039](https://github.com/user-attachments/assets/23e32f63-60ea-4f49-a1ce-45337c2f31a8)

```
chi2,p, _, _ = chi2_contingency(contigency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![Screenshot 2025-04-22 112126](https://github.com/user-attachments/assets/45a0254c-ae06-4320-b6bf-fda6683fa49c)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest , mutual_info_classif,f_classif
data={
'Feature1':[1,2,3,4,5],
'Feature2':['A','B','C','A','B'],
'Feature3':[0,1,1,0,1],
'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![Screenshot 2025-04-22 112206](https://github.com/user-attachments/assets/f12116a7-6b10-4600-9964-c0d594ac5211)


# RESULT:
```
Thus we performed Feature Scaling and Feature Selection process
```
