pip install ydata-profiling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = pd.read_csv("D:/DATA SCIENCE PROGRAMS/DATA SETS/Hospitals.csv")
data.dtypes
data.info()
data.describe()
data.head()
data.isna().sum()
### ANY DUPLICATED VALUES #####
duplicate=data.duplicated()
duplicate
sum(duplicate)
data.min()## MIN VALUES ###
data.max()
data.count()
data.mean()### MEAN VALUES ####
data.median()
data.mode()
data.var()## VARIANCE ####
data.std()### TOTAL VALUES ####
data.skew()
data.kurt()

##### Out layers #######
data = pd.read_csv("D:/DATA SCIENCE PROGRAMS/DATA SETS/Hospitals.csv")
data.dtypes
sns.boxplot(data["Primary Health Centres"])

IQR = data['Primary Health Centres'].quantile(0.75) - data['Primary Health Centres'].quantile(0.25)
lower_limit = data['Primary Health Centres'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Primary Health Centres'].quantile(0.75) + (IQR * 1.5)

#pip install feature_engine
#conda install -c conda-forg feature_engine
from feature_engine.outliers import Winsorizer
Winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['Primary Health Centres'])
data_t =Winsor.fit_transform(data[['Primary Health Centres']])
sns.boxplot(data_t["Primary Health Centres"])


data.dtypes
sns.boxplot(data["Community Health Centres"])

IQR = data['Community Health Centres'].quantile(0.75) - data['Community Health Centres'].quantile(0.25)
lower_limit = data['Community Health Centres'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Community Health Centres'].quantile(0.75) + (IQR * 1.5)

#pip install feature_engine
#conda install -c conda-forg feature_engine
from feature_engine.outliers import Winsorizer
Winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['Community Health Centres'])
data_t =Winsor.fit_transform(data[['Community Health Centres']])
sns.boxplot(data_t["Community Health Centres"])


## LABEL ENCODING OBJECTED COLUM ####

data.head()
data.tail()
data.dtypes

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
print(label_encoder)
data["Hospitals"] = label_encoder.fit_transform(data["Hospitals"])
data.dtypes

#### REPORTING ##### OVERVIEW ####
from ydata_profiling import ProfileReport
report = ProfileReport(data,explorative=True)
report
report.to_file("Hospitaldata.html")
os.getcwd()
