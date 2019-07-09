import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import cv2


img = cv2.imread(r'/home/poornachand/Pictures/data.png',cv2.IMREAD_GRAYSCALE)
plt.imshow(img,cmap = plt.cm.binary_r)
plt.title("This is the first 5 rows of our raw Data...")
plt.show()
img = cv2.imread(r'/home/poornachand/Pictures/data2.png',cv2.IMREAD_GRAYSCALE)
plt.imshow(img,cmap = plt.cm.binary_r)
plt.title("This is the first 5 rows of our Data after preprocessing...\nAll the Strings are converted to Numbers to directly feed our Model..")
plt.show()
img = cv2.imread(r'/home/poornachand/Pictures/data3.png',cv2.IMREAD_GRAYSCALE)
plt.imshow(img,cmap = plt.cm.binary_r)
plt.title("This is the first 5 rows of our Data After Dimentionality reduction using PCA...")
plt.show()

def add_Dimention(X,y):
    X1 = X[:,0]
    X2 = X[:,1]
    X3 = (((X1+X2)**2)*y) + (2*y)
    X_new = np.zeros((X.shape[0],3))
    X_new[:,:-1] = X
    X_new[:,-1] = X3
    return X_new


def plot3D(X,y,show=True):
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection='3d')
    X1=X[:,0]
    X2=X[:,1]
    X3=X[:,2]
    
    ax.scatter(X1,X2,X3,zdir='z',s=20,c=y,depthshade=True)
    plt.title("This is how Our Data Looks in higher Dimention...")
    if (show==True):
        plt.show()
    
    return ax


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
X_train_direct, X_test_direct, y_train_direct, y_test_direct = train_test_split(X,y,test_size=0.2)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf["Class"] = y


X_train_with_PCA, X_test_with_PCA, y_train_with_PCA, y_test_with_PCA = train_test_split(principalDf.drop(["Class"],axis=1).values,y)

plt.scatter(X_train_with_PCA[:,0],X_train_with_PCA[:,1],c=y_train_with_PCA)

plt.scatter(X_test_with_PCA[:,0],X_test_with_PCA[:,1],c=y_test_with_PCA)
plt.title("This is how Our Data Looks when its dimentions reduced by PCA...\n12 Dimentions Reduced to 2 Dimentions")
plt.show()


Scaled_X = StandardScaler().fit_transform(X)
Scaled_X = pca.fit_transform(Scaled_X)


X_train_with_scaled_PCA, X_test_with_scaled_PCA, y_train_with_scaled_PCA, y_test_with_scaled_PCA = train_test_split(Scaled_X,y,test_size=0.2)

plt.scatter(X_train_with_scaled_PCA[:,0],X_train_with_scaled_PCA[:,1],c=y_train_with_scaled_PCA)
plt.title("This is how Our Data Looks when we Apply Feature Scaling")
plt.show()

new_X_train_with_scaled_PCA = add_Dimention(X_train_with_scaled_PCA,y_train_with_scaled_PCA)
new_X_test_with_scaled_PCA = add_Dimention(X_test_with_scaled_PCA,y_test_with_scaled_PCA)

plot3D(new_X_train_with_scaled_PCA,y_train_with_scaled_PCA)

scores = []
model_direct = LogisticRegression()
model_direct.fit(X_train_direct,y_train_direct)
scores.append(model_direct.score(X_test_direct,y_test_direct))

model_with_PCA = LogisticRegression()
model_with_PCA.fit(X_train_with_PCA,y_train_with_PCA)
scores.append(model_with_PCA.score(X_test_with_PCA,y_test_with_PCA))

model_with_scaled_PCA = LogisticRegression()
model_with_scaled_PCA.fit(X_train_with_PCA,y_train_with_PCA)
scores.append(model_with_scaled_PCA.score(X_test_with_PCA,y_test_with_PCA))


model_with_highDimention = LogisticRegression()
model_with_highDimention.fit(new_X_train_with_scaled_PCA,y_train_with_scaled_PCA)
scores.append(model_with_highDimention.score(new_X_test_with_scaled_PCA,y_test_with_scaled_PCA))

plt.barh(["Model With Direct Values","Model With PCA","Model with Scaled PCA","Model in Higher Dimention"],scores,color="bgyr")
plt.title("Comparision of Our Model Performance on test data...")
plt.show()
