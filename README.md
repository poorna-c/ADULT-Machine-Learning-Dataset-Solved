## ADULT-Machine-Learning-Dataset-Solved-100-Acc
ADULT Dataset (https://archive.ics.uci.edu/ml/datasets/Adult) Solved by projecting into higher dimentions with 100% Accuracy

Steps:
1) Reading the Data and loading into memory
2) Preprocesssing the data
3) Applying Feature scaling on the Data
4) Dimentionality Reduction Using PCA
5) Higher Dimention Projection
6) Evaluating model preformance, Train,Test and Deploy.

### 1. Reading the Data and loading into memory
![data](https://user-images.githubusercontent.com/46214838/60877426-6adb6e80-a25b-11e9-80d7-59eed3214153.png)
  The Above image shows the first 5 rows of our data. We can see the input data also contain strings. Hence the data need to be processed before feeding to our model.
  
### 2. Preprocessing the data
![data2](https://user-images.githubusercontent.com/46214838/60877685-ec330100-a25b-11e9-899c-a77f64a15970.png)
  Here I have dropped two columns "native-country" and "relationship" and in all the remaining columns strings are converted into numbers, so that we can directly feed the data to model.(Anyway we are not going to feed directly here)
  
### 3.Applying Feature Scaling on the Data
After applying Feature Scaling on the Data, the data will look like :
![data1](https://user-images.githubusercontent.com/46214838/60878479-69ab4100-a25d-11e9-80ad-d1d01c597b26.png)
  
### 4.Dimentionality Readuction Using PCA
  I have reduced the dimentions to 2 so that we can plot the data on the graph. The Data after reducing the dimentions will look like :
  ![data3](https://user-images.githubusercontent.com/46214838/60878788-0ff74680-a25e-11e9-9483-eb01eba50639.png)
  
### 5.Higher Dimention Projection
  This is how our data looks:
  #### 1)Before Feature Scaling.
  ![Img1](https://user-images.githubusercontent.com/46214838/60879015-767c6480-a25e-11e9-9016-4b04dde9d309.png)
  
  #### 2)After Applying Feature Scaling :
  ![img2](https://user-images.githubusercontent.com/46214838/60879218-d83cce80-a25e-11e9-8f23-f86e6c832778.png)
  #### 3)After Projecting the data to the higher Dimention
![Final Img](https://user-images.githubusercontent.com/46214838/60877071-b3def300-a25a-11e9-8aac-5ae30ed025de.png)


### 6. Evaluating model preformance
  The Following image shows the model performance on the dataset 
    Without any Scaling and Dimentionality Readuction VS Scaled VS Dimentional Reduction VS Higher Dimention 
    ![img3](https://user-images.githubusercontent.com/46214838/60879691-bf80e880-a25f-11e9-92f8-2c5b8e7dde33.png)
