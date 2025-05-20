from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import joblib
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import socket 
from threading import Thread 
from socketserver import ThreadingMixIn
import pickle
from PIL import Image, ImageTk
import tkinter as tk



main = tkinter.Tk()
main.title("Automated Star Type Classification System") #designing main screen
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")
# Load and set background image
bg_image_path = "background.png"  # Replace with your image file path
bg_image = Image.open(bg_image_path)
bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)
# Create a background label
bg_label = tk.Label(main, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)


global filename, df
accuracy = []
precision = []
recall = []
fscore = []
global  X_train, X_test, y_train, y_test
global classifier
global x, y, sc
labels = ['Red Dwarf','Brown Dwarf','White Dwarf','Main Sequence','Super Gaints','Hyper Gaints']

def upload():
    global filename, df, mydict
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' dataset loaded\n')
    df = pd.read_csv(filename)
    
    mydict = {0:"Red Dwarf",1: "Brown Dwarf", 2: "White Dwarf", 3: "Main Sequence"
          , 4: "Super Giants", 5: "Hyper Giants"}
    for i in tqdm(range(len(mydict.keys()))):
        df["Type"] = df["Type"].replace(i,mydict[i])
              
    text.insert(END,str(df.head())+"\n")
    text.insert(END,"Dataset contains total sample records    : "+str(df.shape[0])+"\n")
    text.insert(END,"Dataset contains total attributes (features) : "+str(df.shape[1])+"\n")
    
    fig = px.bar(df, x = df.Type.value_counts().keys(), y = list(df.Type.value_counts()), color = df.Type.value_counts().keys(), 
             title="Count of each data")
    fig.show()
    
    #label = dataset.groupby('Type').size()
    #label.plot(kind="bar")
    #plt.show()

def processDataset():
    global X_train, X_test, y_train, y_test, scaler, label_encoders
    global x, y
    global df
    text.delete('1.0', END)
    
    # Apply label encoding to all object (categorical) columns
    label_encoders = {}  # Dictionary to store label encoders for each column
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoder if needed later
    dat = df.drop(["Type"],axis=1)
    x,y = dat,df["Type"]
    #standard scalar
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    text.insert(END,"Dataset Preprocessing, Normalizing & Shuffling Task Completed\n")
    text.insert(END,str(x)+"\n\n")

    #train test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    text.insert(END,"Data Splitting deatils \n\n")
    text.insert(END,"Training Data : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Testing Data : "+str(X_test.shape[0])+"\n")

def perform_eda():
    """
    Performs Exploratory Data Analysis (EDA) by generating 10 different plots based on the target class `Type`.
    """
    global df

    # Set style
    sns.set(style="whitegrid")
    

    # 2. Correlation heatmap
    plt.figure(figsize=(10, 6))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()
    
    # 3. Boxplot for Temperature across Type
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Type', y='Temperature', data=df)
    plt.title("Boxplot of Temperature across Types")
    plt.show()
    
    # 4. Countplot for categorical variables (Color vs Type)
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Color', hue='Type', data=df)
    plt.xticks(rotation=45)
    plt.title("Distribution of Color by Type")
    plt.show()

    
    # 7. Scatter plot of Temperature vs Absolute Magnitude (A_M) colored by Type
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Temperature', y='A_M', hue='Type', data=df, palette='deep')
    plt.title("Scatter Plot of Temperature vs Absolute Magnitude")
    plt.show()
    
    # 8. Boxen plot for Spectral Class across Type
    plt.figure(figsize=(8, 5))
    sns.boxenplot(x='Type', y='Spectral_Class', data=df)
    plt.title("Boxen Plot of Spectral Class by Type")
    plt.show()
    
    # 9. Bar plot for mean Temperature by Type
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Type', y='Temperature', data=df, estimator=np.mean, ci='sd')
    plt.title("Mean Temperature by Type")
    plt.show()
    
    # 10. Pairwise relationships between numerical columns using pairplot (hue=Type)
    sns.pairplot(df, hue='Type', diag_kind='hist')
    plt.suptitle("Pairwise Relationships Between Numerical Features", y=1.02)
    plt.show()
    
#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, testY,predict):
    global labels
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")        
    report = classification_report(predict, testY,target_names=labels)
    text.insert(END, algorithm+" Classification report\n" +str(report)+"\n")
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    



def runLRC():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    global X_train, X_test, y_train, y_test


    if os.path.exists('model/LogisticRegression.pkl'):
        # Load the trained model from the file
        clf = joblib.load('model/LogisticRegression.pkl')
        predict = clf.predict(X_test)
        calculateMetrics("Logistic Regression", predict, y_test)
    else:
        # Train the model (assuming X_train and y_train are defined)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(clf, 'model/LogisticRegression.pkl')
        predict = clf.predict(X_test)
        calculateMetrics("Logistic Regression Classifier", predict, y_test)
        
def runNB():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    global X_train, X_test, y_train, y_test


    if os.path.exists('model/naive_bayes_model.pkl'):
        # Load the trained model from the file
        clf = joblib.load('model/naive_bayes_model.pkl')
        predict = clf.predict(X_test)
        calculateMetrics("Naive Bayes Classifier", predict, y_test)
    else:
        # Train the model (assuming X_train and y_train are defined)
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(clf, 'model/naive_bayes_model.pkl')
        predict = clf.predict(X_test)
        calculateMetrics("Naive Bayes Classifier", predict, y_test)

def runKNN():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, clf
    # Check if the model files exist
    if os.path.exists('model/KNeighborsClassifier.pkl'):
        # Load the trained model from the file
        clf = joblib.load('model/KNeighborsClassifier.pkl')
        print("Model loaded successfully.")
        predict = clf.predict(X_test)
        calculateMetrics("KNN Classifier", predict, y_test)
    else:
        clf= KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(clf, 'model/KNeighborsClassifier.pkl') 
        print("Model saved successfuly.")
        predict = clf.predict(X_test)
        calculateMetrics("KNN Classifier", predict, y_test)
    
def runRFC():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, mlp
    # Check if the model files exist
    if os.path.exists('model/RandomForestClassifier.pkl'):
        # Load the trained model from the file
        mlp = joblib.load('model/RandomForestClassifier.pkl')
        predict = mlp.predict(X_test)
        calculateMetrics("Random Forest Classifier", predict, y_test)
    else:
        mlp = RandomForestClassifier()
        mlp.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(mlp, 'model/RandomForestClassifier.pkl') 
        predict = mlp.predict(X_test)
        calculateMetrics("Random Forest Classifier", predict, y_test)
   

def graph():
    df = pd.DataFrame([['GNB Model','Precision',precision[0]],['GNB Model','Recall',recall[0]],['GNB Model','F1 Score',fscore[0]],['GNB Model','Accuracy',accuracy[0]],
                       ['KNN Model','Precision',precision[1]],['KNN Model','Recall',recall[1]],['KNN Model','F1 Score',fscore[1]],['KNN Model','Accuracy',accuracy[1]],
                       ['RFC Model','Precision',precision[2]],['RFC Model','Recall',recall[2]],['RFC Model','F1 Score',fscore[2]],['RFC Model','Accuracy',accuracy[2]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()


def StarPrediction():
    text.delete('1.0', END)
    global mlp, test1, predict, scaler, label_encoders
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    test = pd.read_csv(filename)
    test1 = test
    #label_encoders = {}  # Dictionary to store label encoders for each column
    # Apply the same label encoding as in the training set
    for col in test.select_dtypes(include=['object']).columns:
        if col in label_encoders:  # Check if the column was encoded in training
            test1[col] = test1[col].map(lambda s: label_encoders[col].transform([s])[0] if s in label_encoders[col].classes_ else -1)
    # Display the transformed test dataset
    test1 = scaler.fit_transform(test1)
    # Make predictions on the selected test data
    predict = mlp.predict(test1)
      
    # Dictionary mapping indices to star types
    mydict = {
        0: "Red Dwarf",
        1: "Brown Dwarf",
        2: "White Dwarf",
        3: "Main Sequence",
        4: "Super Giants",
        5: "Hyper Giants"
    }

    # Print header
    text.insert(END, f'Predicted Outcomes for each row:\n')

    # Iterate through each row of the dataset and print its corresponding predicted outcome
    for index, row in test.iterrows():
        # Get the prediction for the current row
        predicted_index = predict[index]
        
        # Map predicted index to its corresponding label using `mydict`
        predicted_outcome = mydict.get(predicted_index, "Unknown")  # Default to "Unknown" if index is not found
        
        # Print the current row of the dataset followed by its predicted outcome
        text.insert(END, f'Row {index + 1}: {row.to_dict()} - Predicted Outcome: {predicted_outcome}\n')

    
  
font = ('times', 18, 'bold')
title = Label(main, text="Automated Star Type Classification with Machine Learning using NASA Data")
title.config(bg='linen', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Dataset", command=upload)
upload.place(x=890,y=100)
upload.config(font=font1)  

processButton = Button(main, text="Preprocessing & Data Splitting", command=processDataset)
processButton.place(x=890,y=150)
processButton.config(font=font1) 


dtButton = Button(main, text="EDA", command=perform_eda)
dtButton.place(x=890,y=200)
dtButton.config(font=font1)


dtButton = Button(main, text="Build & Train LRC Model", command=runLRC)
dtButton.place(x=890,y=250)
dtButton.config(font=font1)


dtButton = Button(main, text="Build & Train GNB Model", command=runNB)
dtButton.place(x=890,y=300)
dtButton.config(font=font1)

rfButton = Button(main, text="Build and Train KNN Model", command=runKNN)
rfButton.place(x=890,y=350)
rfButton.config(font=font1)

knnButton = Button(main, text="Build & Train RFC model", command=runRFC)
knnButton.place(x=890,y=400)
knnButton.config(font=font1)

graphButton = Button(main, text="Performance Evaluation", command=graph)
graphButton.place(x=890,y=450)
graphButton.config(font=font1)

predictButton = Button(main, text="Star Prediction on Test Data", command=StarPrediction)
predictButton.place(x=890,y=650)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text = Text(main,height=30,width=100)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='misty rose')
main.mainloop()
