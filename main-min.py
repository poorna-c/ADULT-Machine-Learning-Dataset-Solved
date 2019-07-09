import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


col_names = ["Age","Workclass","fnlwgt","education","education-num","marital-status","occupation","relationship",
            "race","sex","capital-gain","capital-loss","hours-per-week",
             "native-country","Class"]

df = pd.read_csv('/home/poornachand/adult_dataset/adult.data',names=col_names)
df.drop(["native-country","relationship"],axis=1,inplace=True)

df["Workclass"] = df["Workclass"].map({' Private':1, ' Self-emp-not-inc':2, ' Local-gov':3, ' ?':9, ' State-gov':4,
       ' Self-emp-inc':8, ' Federal-gov':7, ' Without-pay':6, ' Never-worked':5})

df["education"] = df["education"].map({' HS-grad':1, ' Some-college':2, ' Bachelors':3, ' Masters':4, ' Assoc-voc':5,
       ' 11th':6, ' Assoc-acdm':7, ' 10th':8, ' 7th-8th':9, ' Prof-school':10, ' 9th':11,
       ' 12th':12, ' Doctorate':13, ' 5th-6th':14, ' 1st-4th':15, ' Preschool':16})

df["marital-status"] = df["marital-status"].map({
    ' Married-civ-spouse':1, ' Never-married':2, ' Divorced':3, ' Separated':4,
       ' Widowed':5, ' Married-spouse-absent':6, ' Married-AF-spouse':7
})

df["race"] = df["race"].map({' White':1, ' Black':2, ' Asian-Pac-Islander':3, ' Amer-Indian-Eskimo':4,
       ' Other':5})

df["sex"]= df["sex"].map({' Male':1, ' Female':2})

df["occupation"] = df["occupation"].map({' Prof-specialty':1, ' Craft-repair':2, ' Exec-managerial':3, ' Adm-clerical':4,
       ' Sales':5, ' Other-service':6, ' Machine-op-inspct':7, ' ?':8,
       ' Transport-moving':9, ' Handlers-cleaners':10, ' Farming-fishing':11,
       ' Tech-support':12, ' Protective-serv':13, ' Priv-house-serv':14,
       ' Armed-Forces':15})

df["Class"] = df["Class"].map({' <=50K':0, ' >50K':1})

X = df.drop(["Class"],axis=1).values
y = df.Class.values

Scaled_X = StandardScaler().fit_transform(X)

Reduced_X = PCA(n_components=2).fit_transform(Scaled_X)

print(Reduced_X)
X1 = Reduced_X[:,0]
X2 = Reduced_X[:,1]
X3 = (((X1+X2)**2)*y) + (2*y)
X_new = np.zeros((X.shape[0],3))
X_new[:,:-1] = Reduced_X
X_new[:,-1] = X3

X_train, X_test, y_train, y_test = train_test_split(X_new,y)

model = LogisticRegression()
model.fit(X_train,y_train)
print("Accuracy : ", model.score(X_test,y_test) * 100 ,"%")

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111,projection='3d')
X1=X_train[:,0]
X2=X_train[:,1]
X3=X_train[:,2]

ax.scatter(X1,X2,X3,zdir='z',s=20,c=y_train,depthshade=True)
plt.title("This is how Our Data Looks in higher Dimention...")
plt.show()