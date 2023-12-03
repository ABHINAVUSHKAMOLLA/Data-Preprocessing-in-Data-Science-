####### DATA PREPROSSING #########

######### 1. TYPE CASTING VARIABLES ##########'
pip install ydata-profiling
import pandas as pd 
import os
data = pd.read_csv("D:/DATA SCIENCE PROGRAMS/DATA SETS/PROJECTS/OLA COUSTOMERS REVIEWS NEW.csv")
data.dtypes
######## 2. Handling duplicates #########
duplicate=data.duplicated()
duplicate
sum(duplicate)

########## 3.OUT LIAYRS ANALYSIS ##########
import numpy as np 
import seaborn as sns
data = pd.read_csv("D:/DATA SCIENCE PROGRAMS/DATA SETS/PROJECTS/OLA COUSTOMERS REVIEWS NEW.csv")
data.dtypes
sns.boxplot(data.rating)
IQR = data['rating'].quantile(0.75) - data['rating'].quantile(0.25)
lower_limit = data['rating'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['rating'].quantile(0.75) + (IQR * 1.5)

#pip install feature_engine
#conda install -c conda-forg feature_engine
from feature_engine.outliers import Winsorizer
Winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['rating'])
data_t =Winsor.fit_transform(data[['rating']])
sns.boxplot(data_t.rating )

################ MISSING VALUES ##########
from sklearn.impute import SimpleImputer  # Import the SimpleImputer class
data.dtypes
data.isna().sum()

most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data["developer_response"] = pd.DataFrame(most_frequent.fit_transform(data[["developer_response"]]))
data["developer_response"].isna().sum()

most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data["developer_response_date"] = pd.DataFrame(most_frequent.fit_transform(data[["developer_response_date"]]))
data["developer_response_date"].isna().sum()

most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data["review_description"] = pd.DataFrame(most_frequent.fit_transform(data[["review_description"]]))
data["developer_response_date"].isna().sum()

most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data["appVersion"] = pd.DataFrame(most_frequent.fit_transform(data[["appVersion"]]))
data["appVersion"].isna().sum()
data.dtypes
######## 6.DESCRITIZATION  ########

data.head()
data.info()
data.describe()
#### BINNING #######
data['rating_new']=pd.cut(data['rating'],
bins=[min(data.rating),
data.rating.mean(),
max(data.rating)],
labels=["Low","High"],
include_lowest=True)

data.head(10)
data.min()
data.max()
data.count()
data.rating_new.value_counts()
data.thumbs_up.value_counts()
data.developer_response.value_counts()
data.appVersion.value_counts()
data.review_id.value_counts()
data.source.value_counts()
data.user_name.value_counts()

########### 7.DUMMY VARIABLES  ##### LABEL ENCODING ##########

import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("D:/DATA SCIENCE PROGRAMS/DATA SETS/PROJECTS/OLA COUSTOMERS REVIEWS NEW.csv")
df.head()
df.tail()
df.dtypes
### Applying the Label Encoding For Targeted Objected Colums #
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df["review_date"] = label_encoder.fit_transform(df["review_date"])
df["source"] = label_encoder.fit_transform(df["source"])
df["user_name"] = label_encoder.fit_transform(df["user_name"])
df["review_description"] = label_encoder.fit_transform(df["review_description"])
df["developer_response"] = label_encoder.fit_transform(df["developer_response"])
df["developer_response_date"] = label_encoder.fit_transform(df["developer_response_date"])
df["appVersion"] = label_encoder.fit_transform(df["appVersion"])
df["laguage_code"] = label_encoder.fit_transform(df["laguage_code"])
df["country_code"] = label_encoder.fit_transform(df["country_code"])
df.head()
df.dtypes #### ALL VALUES ARE CONVERTED IN TO INT VALUES AS PER DATASET######

#######  8.Standadization And Normalization   #############
import pandas as pd 
import numpy as np 
###### STANDARDIZATION  #######
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("D:/DATA SCIENCE PROGRAMS/DATA SETS/PROJECTS/OLA COUSTOMERS REVIEWS NEW.csv")
a = data.describe()
scaler = StandardScaler()
df = scaler.fit_transform(a)
dataset = pd.DataFrame(df)
res = dataset.describe()
data.columns 

data.drop(['source', 'review_id', 'user_name'], axis=1, inplace=True)
a1 = data.describe()

######## GET DUMMIES #######

data = pd.get_dummies(data, drop_first = True)
data

######### 9.EXPLORARAY DATA ANALYSIS MOMENTS #######

import pandas as pd
ola = pd.read_csv("D:/DATA SCIENCE PROGRAMS/DATA SETS/PROJECTS/OLA COUSTOMERS REVIEWS NEW.csv")
ola.info()
ola.thumbs_up.mean()
ola.thumbs_up.median()
ola.thumbs_up.mode()
######## Second Moment Method #########
ola.thumbs_up.var()
ola.thumbs_up.std()

####### Third moment method ########
ola.thumbs_up.skew()
ola.thumbs_up.std()
####Fourth moment #######
ola.thumbs_up.kurt()
ola.thumbs_up.skew()

###### Grapical Representation ########
#pip install ydata-profiling
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ydata_profiling import ProfileReport
import os
ola = pd.read_csv("D:/DATA SCIENCE PROGRAMS/DATA SETS/PROJECTS/OLA COUSTOMERS REVIEWS NEW.csv")
ola.shape
ola.dtypes

plt.bar(height = ola.rating, x = np.arange(1, 1000)) 
plt.title("rating")


plt.hist(ola.thumbs_up, color = 'red')
plt.title("thums_up")


plt.stem(ola.thumbs_up)
plt.title("thums_up")

plt.boxplot(ola.thumbs_up)
plt.stem(ola.thumbs_up)

#### REPORTING ##### OVERVIEW ####
report = ProfileReport(ola,explorative=True)
report
report.to_file("EDA_report.html")

os.getcwd()

















































