import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score

dataset=pd.read_csv('diabetes.csv')

# print(dataset.head())
print(dataset.shape)
# print(dataset.describe())
# print(dataset['Outcome'].value_counts())
# print(dataset.groupby('Outcome').mean())

#seperating data and labels
X=dataset.drop(columns='Outcome',axis=1)
Y=dataset["Outcome"]
print(X)
# print(Y)

# standardisation
scaler=StandardScaler()
scaler.fit(X)
std_data=scaler.transform(X)
print(std_data)
X=std_data

# print(X)
# print(Y)
#splitting training and training set
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=3)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#training the model
classifier=svm.SVC(kernel='linear')

#training the support vector machine classifier
print(classifier.fit(x_train,y_train))
#model evaluation

x_train_accuracy=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_accuracy,y_train)
print("accuracy score of training data:",training_data_accuracy)

x_test_accuracy=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_accuracy,y_test)
print("accuracy score of test data:",test_data_accuracy)

#making a prediction
input_data=(10,100,88,60,110,46.8,0.942,31)
#changing the input data to numpy array

numpy_array=np.asarray(input_data)
#reshape the array
reshaped=numpy_array.reshape(1,-1)
std_data=scaler.transform(reshaped)
# print(std_data)
prediction=classifier.predict(std_data)
# print(prediction)

if(prediction[0]==0):
    print("the person is not diabetic")
else:
    print("the person is diabetic")





# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.metrics import accuracy_score

# class DiabetesPredictor:
#     def __init__(self, dataset_path):
#         self.dataset_path = dataset_path
#         self.model = svm.SVC(kernel='linear')
#         self.scaler = StandardScaler()

#     def load_data(self):
#         dataset = pd.read_csv(self.dataset_path)
#         X = dataset.drop(columns='Outcome', axis=1)
#         Y = dataset['Outcome']
#         return X, Y

#     def preprocess_data(self, X):
#         self.scaler.fit(X)
#         return self.scaler.transform(X)

#     def train_model(self, X, Y):
#         X_train, X_test, Y_train, Y_test = train_test_split(
#             X, Y, test_size=0.2, stratify=Y, random_state=2
#         )
#         self.model.fit(X_train, Y_train)

#         # Evaluate the model
#         train_accuracy = accuracy_score(Y_train, self.model.predict(X_train))
#         test_accuracy = accuracy_score(Y_test, self.model.predict(X_test))

#         return train_accuracy, test_accuracy

#     def predict(self, input_data):
#         input_array = np.asarray(input_data).reshape(1, -1)
#         standardized_data = self.scaler.transform(input_array)
#         prediction = self.model.predict(standardized_data)
#         return 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

# if __name__ == "__main__":
#     # Initialize predictor
#     predictor = DiabetesPredictor('diabetes.csv')

#     # Load and preprocess data
#     X, Y = predictor.load_data()
#     X = predictor.preprocess_data(X)

#     # Train the model
#     train_accuracy, test_accuracy = predictor.train_model(X, Y)
#     print(f"Training Accuracy: {train_accuracy:.2f}")
#     print(f"Testing Accuracy: {test_accuracy:.2f}")

#     # Make a prediction
#     input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
#     result = predictor.predict(input_data)
#     print(f"Prediction for input data: {result}")
