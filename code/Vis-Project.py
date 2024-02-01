'''
DATS 6401 Visualization of Complex Data - Lab5
Name: Aihan Liu
GWID: G45894738
Date: 4/2/2022
'''

from dash import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import pandas as pd
import math
import plotly.graph_objects as go
from datetime import date
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

### READ DATA
PATH = os.getcwd()
print(PATH)
DATA_DIR = os.getcwd() + os.path.sep + 'PRSA_Data_20130301-20170228' + os.path.sep

if not os.path.exists(DATA_DIR + 'ALL.csv'):
    frames = []
    for file in os.listdir(DATA_DIR):
        if file[-4:] == '.csv':
            FILE_NAME = DATA_DIR + os.path.sep + file
            dataframe = pd.read_csv(FILE_NAME, header=0)
            frames.append(dataframe)

    result = pd.concat(frames)
    '''
    check missing values
    '''
    df_orig = result.dropna(axis=0, how='any')

    '''
    outliers
    '''
    colindex = list(np.arange(5, 15))
    colindex.append(16)
    df = df_orig[(np.abs(stats.zscore(df_orig.iloc[:, colindex])) < 3).all(axis=1)]
    print(len(df))
    print(f'The number of outliers is {len(df_orig) - len(df)}')

    '''
    Format Date
    '''
    cols = ["year", "month", "day", "hour"]
    df['date'] = df[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d-%H')
    df['date_YM'] = df['year'] + df['month'] / 12

    df.to_csv(DATA_DIR + 'ALL.csv', index=False)
    print('Data merged!')
else:
    df = pd.read_csv(DATA_DIR + 'ALL.csv', header=0)
    print('Data read!')


colnames = df.columns
print(len(df))
print(colnames)
print(df.head())


'''
PCA
'''
numeric_col = df[df._get_numeric_data().columns.to_list()[6:]]
X = StandardScaler().fit_transform(numeric_col)

# singular values and conditional number for original feature space
H = np.matmul(X.T, X)
_, d, _ = np.linalg.svd(H)
print(f'Original Data: singular Values {d}')
# minimal singular value is 11.69699391
print(f'Original Data: condition number {LA.cond(X)} ')

pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
X_PCA = pca.transform(X)
print('Original Dim', X.shape)
print('Transformed Dim', X_PCA.shape)
print(f'explained variance ratio {pca.explained_variance_ratio_}')

H_reduce = np.matmul(X_PCA.T, X_PCA)
_, d, _ = np.linalg.svd(H_reduce)
print(f'Reduced Dimensions Data: singular Values {d}')
# minimal singular value is 11.69699391
print(f'Reduced Dimensions Data: condition number {LA.cond(X_PCA)} ')

# cumulative explained variance
plt.figure()
l = np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1, 1)
plt.xticks(l)
plt.plot(l, np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.grid()
plt.show()

'''
Normality Test
'''
alpha = 0.05
colindex = list(np.arange(5, 15))
colindex.append(16)
# q-q plot
fig, ax = plt.subplots(4, 3, figsize=(10, 14))
i = 0
j = 0
for idx in colindex:
    data = np.array(df.iloc[:, idx])
    names = colnames[idx]
    sm.qqplot(data, ax=ax[i, j], line='q')
    ax[i, j].set_title(names)
    ax[i, j].grid()
    if (i+1) % 4 == 0 and i != 0:
        i = 0
        j += 1
    else:
        i += 1
plt.tight_layout()
plt.show()

for idx in colindex:
    data = df.iloc[:, idx]
    names = colnames[idx]
    norm_test = stats.kstest(data, 'norm')
    print(f'{names} K-S test: statistics= {norm_test[0]:.3f} p-value = {norm_test[1]:.2f}.')
    if norm_test[1] > alpha:
        print(f'{names} is normal in the {alpha} level.')
    else:
        print(f'{names} is not normal in the {alpha} level.')

'''
Heatmap
'''
pd.options.display.float_format = "{:,.2f}".format
pd.set_option('display.max_columns', None)
numeric_col = df[df._get_numeric_data().columns.to_list()[5:-1]]
numeric_df = pd.DataFrame(numeric_col)
corr_numeric = numeric_df.corr()
plt.figure(figsize=(10,9))
heatmap = sns.heatmap(corr_numeric, annot=True, cmap="rocket", fmt='.2g')
plt.title('Correlation Coefficient between features')
plt.show()

# numeric_col2 = df[df._get_numeric_data().columns.to_list()[5:11]]
# numeric_df2 = pd.DataFrame(numeric_col2)
# plt.figure(figsize=(10,9))
# # sns.pairplot(numeric_df)
# pd.plotting.scatter_matrix(numeric_df2, alpha=0.2)
# plt.title('Correlation Coefficient between features')
# plt.show()


'''
Seaborn visualization
'''
df_2017 = df[(df['year'] == 2017) & (df['month'] == 2)]
df_2017_urban = df_2017.loc[df_2017['station'].isin(['Dongsi', 'Wanliu', 'Aotizhongxin', 'Gucheng'])]
# line plot
sns.set_theme(style='darkgrid')

plt.figure(figsize=(9,5))
sns.lineplot(data=df_2017_urban,
             x = 'day',
             y = 'PM2.5',
             hue = 'station',
             ci= None)
plt.show()

sns.countplot(data=df_2017_urban,
              x = 'PM2.5',
              hue = 'station',
              palette="husl",)
plt.show()

plt.figure(figsize=(9,5))
sns.histplot(data=df_2017_urban, x="PM2.5", hue="station", multiple = 'stack')
plt.show()

plt.figure(figsize=(9,5))
sns.histplot(data=df_2017_urban, x="PM2.5", hue="station", multiple = 'dodge')
plt.show()

# count plot
plt.figure(figsize=(9,5))
sns.countplot(x="wd", data=df_2017_urban)
plt.show()

plt.figure(figsize=(10, 5))
sns.catplot(x="day", y="PM2.5", hue="station", data=df_2017_urban)
plt.show()

# pie plot
plt.figure(figsize=(10, 5))
wind = df_2017_urban['wd'].value_counts()
score = list(wind)
label = list(wind.index)
explode = (.03, .03, .03, .03, .03, .03, .03, .03, .03, .03, .03, .03, .03, .03, .03, .03)
fig, ax = plt.subplots(1,1)
ax.pie(score,  labels=label, explode = explode, autopct='%1.1f%%')
ax.axis('square')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set_title('Pie chart for different direction of wind')
plt.tight_layout()
plt.show()

# displot
plt.figure(figsize=(9,5))
sns.displot(data=df_2017_urban, x="PM2.5", col="station")
plt.show()

# pairplot
collist = list(np.arange(5, 11))
collist.append(17)
df_17_num = df_2017_urban[df_2017_urban.columns.to_list()[11:18]]
plt.figure(figsize=(10, 10))
sns.pairplot(df_17_num, hue="station")
plt.show()

# Kernal density estimate
plt.figure(figsize=(9,5))
sns.displot(data=df_2017_urban, x="PM2.5",hue="station",  kind="kde")
plt.show()


# Scatter plot and regression line using sklearn
x = df_2017_urban['CO'].values[:,np.newaxis]
y = df_2017_urban['PM2.5'].values
model2 = LinearRegression()
model2.fit(x, y)

plt.scatter(x, y,color='g')
plt.plot(x, model2.predict(x), color='k')
plt.xlabel('CO')
plt.ylabel('PM2.5')
plt.title('CO vs PM2.5')
plt.show()

# boxplot
plt.figure(figsize=(13,5))
sns.boxplot(x="station", y="PM2.5", data=df_2017)
plt.show()

# violin plot
plt.figure(figsize=(13,5))
sns.violinplot(x="station", y="PM2.5", data=df_2017)
plt.show()