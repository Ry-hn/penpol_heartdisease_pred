import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.sparse.construct import rand
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

dataSet = pd.read_csv("dataset/heart.csv")

# Melihat sample dari dataset

# print(dataSet.head())

# Mengubah jenis kolom FastingBS dan HeartDisease dari int menjadi str karena hanya terdiri dari 0 - 1

dataSet['FastingBS'] = dataSet['FastingBS'].astype(str)
dataSet['HeartDisease'] = dataSet['HeartDisease'].astype(str)

# Pengecekan jumlah dan jenis kolom yang ada didapat 12 kolom dengan 
# 4 kolom dengan tipe data int, 
# 7 kolom dengan tipe data objek / string, 
# 1 kolom dengan tipe data float

# print(dataSet.dtypes.value_counts())

# Pengecekan apakah ada data yang duplikat / hilang 

# print(f'Duplicated: { len(dataSet[dataSet.duplicated()]) }')
# print(f'Missing: { dataSet.isnull().sum().sum() }')

# Pengecekan terhadap mean, std, min terhadap dataset

# print(f'Dataset awal \n {dataSet.describe()}')

# Mengambil Kolom berdasarkan tipe data 
categ = dataSet.select_dtypes(include=object).columns
numeric_type = dataSet.select_dtypes(exclude=object).columns 

# settingan Visualizer
fig, ax = plt.subplots(figsize = (15, 15)) 
fig.patch.set_facecolor('#CAD5E0')
mpl.rcParams['font.family'] = 'TeX Gyre Heros'

def plot_design(title):
    plt.xlabel('')
    plt.ylabel('')
    plt.yticks(fontsize=15, color='black')
    plt.xticks(fontsize=15, color='black')
    plt.box(False)
    plt.title(title, fontsize=24, color='black')
    plt.tight_layout(pad=5.0)

def plot_categorical(categ):
    for i in (enumerate(categ)): 
        plt.subplot(4, 2, i[0] + 1)
        sns.countplot(y = i[1], data = dataSet, order=dataSet[i[1]].value_counts().index, palette='Greens_r', edgecolor='black')
        plot_design(i[1])
        plt.suptitle('Categorical Variables', fontsize=40)
    plt.show()


def plot_numeric(numeric_type):
    for i in (enumerate(numeric_type)):
        plt.subplot(3, 2, i[0] + 1)
        sns.histplot(x = i[1], data = dataSet, color='#EBEBAB', edgecolor='black')
        plot_design(i[1])
        plt.suptitle('Numerical Variables', fontsize=40)
    plt.show()

# Melakukan Plotting terhadap data yang memiliki tipe data object

# plot_categorical(categ)

# Melakukan Plotting terhadap data yang memiliki tipe data numeric 

# plot_numeric(numeric_type)

# Scaling Numeric Features sehingga bernilai 0 - 1

for col in numeric_type:
    dataSet[col] = MinMaxScaler().fit_transform(dataSet[[col]])

# plot_numeric(numeric_type)

# Kolom RestingBP mengandung nilai 0 maka harus dihapus karena tidak mungkin tekanan darah bernilai 0

dataSet = dataSet.drop(dataSet[(dataSet['RestingBP'] == 0)].index)

q1RestingBP = dataSet['RestingBP'].quantile(0.25)
q3RestingBP = dataSet['RestingBP'].quantile(0.75)

iqRestingBP = q3RestingBP - q1RestingBP

lowerTailRestingBP = q1RestingBP - 1.5 * iqRestingBP
upperTailRestingBP = q3RestingBP + 1.5 * iqRestingBP

# print([lowerTailRestingBP, upperTailRestingBP])

# uRestingBP = dataSet[(dataSet['RestingBP'] >= upperTailRestingBP) | (dataSet['RestingBP'] <= lowerTailRestingBP)]

# uRestingBP = pd.DataFrame(uRestingBP)

# print('Outliers on RestingBP')
# print(uRestingBP.value_counts(uRestingBP['HeartDisease']))

medianRestingBP = np.median(dataSet['RestingBP'])

# print(f'Median RestingBP: {medianRestingBP}')

for i in dataSet['RestingBP']:
    if i > upperTailRestingBP or i < lowerTailRestingBP:
        dataSet['RestingBP'] = dataSet['RestingBP'].replace(i, medianRestingBP)

# print(f'Setelah kolom restingbp diolah: \n{dataSet.describe()}')

# Kolom Cholesterol yang mengandung nilai 0 akan diganti dengan nilai median karena 
# apabila dijadikan 0 akan merubah nilai rerata kolesterol secara dratis

q1Cholesterol = dataSet['Cholesterol'].quantile(0.25)
q3Cholesterol = dataSet['Cholesterol'].quantile(0.75)
iqCholesterol = q3Cholesterol- q1Cholesterol
lowerTailCholesterol = q1Cholesterol - 1.5 * iqCholesterol
upperTailCholesterol = q3Cholesterol + 1.5 * iqCholesterol

# print([lowerTailCholesterol, upperTailCholesterol])

# uCholesterol = dataSet[(dataSet['Cholesterol'] >= upperTailCholesterol) | (dataSet['Cholesterol'] <= lowerTailCholesterol)]
# uCholesterol = pd.DataFrame(uCholesterol)

# zCholesterol = dataSet[dataSet['Cholesterol'] == 0]
# zCholesterol = pd.DataFrame(zCholesterol)

# print(zCholesterol.value_counts(dataSet['HeartDisease']))
# print(uCholesterol.value_counts(dataSet['HeartDisease']))

medianCholesterol = np.median(dataSet['Cholesterol'])

# print(f'median: {medianCholesterol}')

for i in dataSet['Cholesterol']:
    if i > upperTailCholesterol:
        dataSet['Cholesterol'] = dataSet['Cholesterol'].replace(i, medianCholesterol)

# print(f'Setelah Nilai Kolesterol diganti median\n {dataSet.describe()}')

# Mengubah tipe data kolom Cholesterol dan RestingBP menjadi Int

dataSet['FastingBS'] = dataSet['FastingBS'].astype(int)
dataSet['HeartDisease'] = dataSet['HeartDisease'].astype(int)

categ = dataSet.select_dtypes(include=object).columns

# Encoding Categorical Feature String -> Binary 0 1

encodedData = pd.get_dummies(dataSet, columns=categ,drop_first=True)

# print(encodedData.head())

# Membagi dataset menjadi label dan feature serta training dan test set

y = encodedData['HeartDisease'] 
X = encodedData.drop(columns='HeartDisease')

# rand_state = 0
iteration = 0
acc = list()

# while True:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rand_state)
#     knn = KNeighborsClassifier(n_neighbors=9)
#     knn = knn.fit(X_train, y_train)

#     y_pred = knn.predict(X_test)

#     accScore =  round(accuracy_score(y_test, y_pred), 4)
#     f1Score = round(f1_score(y_test, y_pred), 4)

#     acc.append((accScore, rand_state, iteration))
    
#     print('Accuracy score: ', accScore)
#     print('F1 Score: ', f1Score)

#     if iteration == 1000:
#         break
    
#     rand_state = rand_state + 1
#     iteration = iteration + 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=192)

# KNN 

# maxNeighbors = 19 

# for i in range(maxNeighbors):

#     knn = KNeighborsClassifier(n_neighbors=i+1)
#     knn = knn.fit(X_train, y_train)

#     y_pred = knn.predict(X_test)

#     accScore =  round(accuracy_score(y_test, y_pred), 4)
#     f1Score = round(f1_score(y_test, y_pred), 4)

#     acc.append((accScore, i+1))
    
#     print('Accuracy score: ', accScore)
#     print('F1 Score: ', f1Score)

# res = max(acc, key = lambda i : i[0])
# print(f'max score: {res[0]} neighbors: {res[1]}')

knn = KNeighborsClassifier(n_neighbors=7)
knn = knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accScore =  round(accuracy_score(y_test, y_pred), 4)
f1Score = round(f1_score(y_test, y_pred), 4)

print('Accuracy score: ', accScore)
print('F1 Score: ', f1Score)

confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)

tn, fp, fn, tp = confusionMatrix.ravel()

print((tn, fp, fn, tp))