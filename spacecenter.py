from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter as tk
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import socket 
from threading import Thread 
from socketserver import ThreadingMixIn
import pickle
from PIL import Image, ImageTk
import tkinter.ttk as ttk

# Constants for UI colors and styling
SPACE_BG = "#0A0F20"        # Dark space blue
STAR_GOLD = "#FFD700"       # Star gold
COSMIC_BLUE = "#4169E1"     # Cosmic blue
NEBULA_PURPLE = "#9370DB"   # Nebula purple
MARS_RED = "#B22222"        # Mars red
TEXT_COLOR = "#FFFFF0"      # Light text color
PANEL_BG = "#1A1F35"        # Slightly lighter than background for panels

# Set up the main application window
main = tk.Tk()
main.title("NASA Star Classification System")
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")
main.configure(bg=SPACE_BG)

# Create a style for the application
style = ttk.Style()
style.theme_use('clam')
style.configure('TButton', 
                background=COSMIC_BLUE, 
                foreground=TEXT_COLOR,
                font=('Arial', 11, 'bold'),
                padding=10,
                borderwidth=2,
                relief='raised')
style.map('TButton', 
          background=[('active', NEBULA_PURPLE), ('pressed', MARS_RED)],
          foreground=[('active', 'white')])

# Load and set space background image
try:
    bg_image_path = "background.jpg"  # Space background image
    bg_image = Image.open(bg_image_path)
    bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    
    # Create a background label
    bg_label = tk.Label(main, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
except Exception as e:
    print(f"Could not load background image: {e}")
    # Create a starry background as fallback
    bg_canvas = Canvas(main, bg=SPACE_BG, highlightthickness=0)
    bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)
    
    # Generate random stars
    for _ in range(200):
        xd = np.random.randint(0, screen_width)
        yd = np.random.randint(0, screen_height)
        size = np.random.randint(1, 3)
        color = np.random.choice(["white", STAR_GOLD, "#FFFFF0"])
        bg_canvas.create_oval(xd, yd, xd+size, yd+size, fill=color, outline="")

# Global variables
global filename, df, x, y, scaler, X_train, X_test, y_train, y_test, classifier, label_encoders, model_used
accuracy = []
precision = []
recall = []
fscore = []
labels = ['Red Dwarf', 'Brown Dwarf', 'White Dwarf', 'Main Sequence', 'Super Giants', 'Hyper Giants']
model_used = None  # Track which model is being used for prediction

# Create a frame for the title with cosmic styling
title_frame = Frame(main, bg=PANEL_BG, bd=2, relief=RIDGE)
title_frame.place(x=screen_width//2 - 450, y=20, width=900, height=80)

# Add a cosmic title with star icon
title_text = "🌟 Automated Star Type Classification System 🌟"
title_label = Label(title_frame, text=title_text, bg=PANEL_BG, fg=STAR_GOLD,
                   font=('Helvetica', 24, 'bold'), pady=10)
title_label.pack(fill=BOTH, expand=True)

# Create a frame for buttons with cosmic styling
button_frame = Frame(main, bg=PANEL_BG, bd=2, relief=RIDGE)
button_frame.place(x=screen_width - 350, y=120, width=320, height=650)

# Button frame title
button_title = Label(button_frame, text="Control Panel", bg=PANEL_BG, fg=STAR_GOLD,
                    font=('Helvetica', 16, 'bold'), pady=10)
button_title.pack(fill=X)

# Create a frame for the text area with cosmic styling
text_frame = Frame(main, bg=PANEL_BG, bd=2, relief=RIDGE)
text_frame.place(x=30, y=120, width=screen_width - 400, height=600)

# Text output area with space styling
text = Text(text_frame, bg="#0A0F20", fg=TEXT_COLOR, padx=10, pady=10,
           font=('Consolas', 12), wrap=WORD)
text.pack(fill=BOTH, expand=True, padx=5, pady=5)

# Add a scrollbar to the text area
scrollbar = Scrollbar(text)
scrollbar.pack(side=RIGHT, fill=Y)
text.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=text.yview)

# Function to create styled buttons
def create_cosmic_button(parent, text, command, icon=None):
    btn_frame = Frame(parent, bg=PANEL_BG, padx=5, pady=5)
    btn_frame.pack(fill=X, padx=20, pady=5)
    
    btn = Button(btn_frame, text=text, command=command, bg=COSMIC_BLUE, 
                fg=TEXT_COLOR, font=('Helvetica', 12, 'bold'), 
                activebackground=NEBULA_PURPLE, activeforeground='white', 
                relief=RAISED, bd=2, padx=10, pady=8, width=25)
    
    # Add hover effect
    def on_enter(e):
        btn['background'] = NEBULA_PURPLE
    def on_leave(e):
        btn['background'] = COSMIC_BLUE
        
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    
    btn.pack(fill=X)
    return btn

# Functions
def upload():
    global filename, df, mydict
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, f"🔭 Dataset loaded: {filename}\n\n")
    df = pd.read_csv(filename)
    
    mydict = {0: "Red Dwarf", 1: "Brown Dwarf", 2: "White Dwarf", 3: "Main Sequence",
              4: "Super Giants", 5: "Hyper Giants"}
    for i in tqdm(range(len(mydict.keys()))):
        df["Type"] = df["Type"].replace(i, mydict[i])
              
    text.insert(END, "📊 Dataset Preview:\n")
    text.insert(END, str(df.head()) + "\n\n")
    text.insert(END, f"📈 Total sample records: {df.shape[0]}\n")
    text.insert(END, f"🔬 Total attributes (features): {df.shape[1]}\n")
    
    counts = df.Type.value_counts().reset_index()
    counts.columns = ['Type', 'Count']
    fig = px.bar(counts, x='Type', y='Count', color='Type', title="Count of Star Types")

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
    dat = df.drop(["Type"], axis=1)
    x, y = dat, df["Type"]
    
    # Standard scalar
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    text.insert(END, "🔄 Dataset Preprocessing Complete\n")
    text.insert(END, "✅ Data Normalized & Shuffled\n\n")
    text.insert(END, str(x[:5]) + "\n...\n\n")  # Show just the first few rows

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    text.insert(END, "📊 Data Split Statistics:\n")
    text.insert(END, f"🚀 Training Data: {X_train.shape[0]} samples\n")
    text.insert(END, f"🧪 Testing Data: {X_test.shape[0]} samples\n")

def perform_eda():
    """
    Performs Exploratory Data Analysis (EDA) by generating different plots based on the target class `Type`.
    """
    global df

    text.delete('1.0', END)
    text.insert(END, "🔍 Performing Exploratory Data Analysis...\n\n")
    
    # Set style
    sns.set(style="dark")
    
    # 1. Correlation heatmap
    text.insert(END, "📊 Generating Correlation Heatmap...\n")
    plt.figure(figsize=(10, 6))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 2. Boxplot for Temperature across Type
    text.insert(END, "📊 Generating Temperature Boxplot...\n")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Type', y='Temperature', data=df, palette='viridis')
    plt.title("Boxplot of Temperature across Types", fontsize=15, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 3. Countplot for categorical variables (Color vs Type)
    text.insert(END, "📊 Generating Color Distribution by Type...\n")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Color', hue='Type', data=df, palette='viridis')
    plt.title("Distribution of Color by Type", fontsize=15, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 4. Scatter plot of Temperature vs Absolute Magnitude
    text.insert(END, "📊 Generating Temperature vs Magnitude Scatter Plot...\n")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Temperature', y='A_M', hue='Type', data=df, palette='viridis', s=100, alpha=0.7)
    plt.title("Temperature vs Absolute Magnitude", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 5. Boxen plot for Spectral Class across Type
    text.insert(END, "📊 Generating Spectral Class Distribution...\n")
    plt.figure(figsize=(10, 6))
    sns.boxenplot(x='Type', y='Spectral_Class', data=df, palette='viridis')
    plt.title("Boxen Plot of Spectral Class by Type", fontsize=15, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 6. Bar plot for mean Temperature by Type
    text.insert(END, "📊 Generating Mean Temperature by Type...\n")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Type', y='Temperature', data=df, estimator=np.mean, errorbar='sd', palette='viridis')
    plt.title("Mean Temperature by Type", fontsize=15, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 7. Pairwise relationships between numerical columns
    text.insert(END, "📊 Generating Pairwise Feature Relationships...\n")
    sns.pairplot(df, hue='Type', diag_kind='kde', palette='viridis')
    plt.suptitle("Pairwise Relationships Between Features", y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    text.insert(END, "✅ EDA Complete! All visualizations have been displayed.\n")
    
def calculateMetrics(algorithm, predict, testY):
    global labels, model_used
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    # Format output with cosmic emojis and better formatting
    text.insert(END, f"🚀 {algorithm} Performance Metrics:\n")
    text.insert(END, f"  ✓ Accuracy:  {a:.2f}%\n")
    text.insert(END, f"  ✓ Precision: {p:.2f}%\n")
    text.insert(END, f"  ✓ Recall:    {r:.2f}%\n")
    text.insert(END, f"  ✓ F1 Score:  {f:.2f}%\n\n")
    
    report = classification_report(testY, predict, target_names=labels)
    text.insert(END, f"📑 {algorithm} Classification Report:\n{report}\n")
    
    # Create a stylized confusion matrix
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize=(8, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, 
                    annot=True, cmap="viridis", fmt="g")
    ax.set_ylim([0, len(labels)])
    plt.title(f"{algorithm} Confusion Matrix", fontsize=15, fontweight='bold') 
    plt.ylabel('True Class', fontsize=12) 
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Store the model name that was last used
    model_used = algorithm

def runLRC():
    text.delete('1.0', END)
    text.insert(END, "🔍 Running Logistic Regression Classifier...\n\n")
    global accuracy, precision, recall, fscore, clf, model_used
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    global X_train, X_test, y_train, y_test

    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    if os.path.exists('model/LogisticRegression.pkl'):
        text.insert(END, "📁 Loading pre-trained Logistic Regression model...\n")
        clf = joblib.load('model/LogisticRegression.pkl')
        predict = clf.predict(X_test)
        calculateMetrics("Logistic Regression", predict, y_test)
    else:
        text.insert(END, "🧠 Training new Logistic Regression model...\n")
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'model/LogisticRegression.pkl')
        predict = clf.predict(X_test)
        calculateMetrics("Logistic Regression", predict, y_test)
    
    # Update the current model for prediction
    model_used = "Logistic Regression"
        
def runNB():
    text.delete('1.0', END)
    text.insert(END, "🔍 Running Gaussian Naive Bayes Classifier...\n\n")
    global accuracy, precision, recall, fscore, clf, model_used
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    global X_train, X_test, y_train, y_test

    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    if os.path.exists('model/naive_bayes_model.pkl'):
        text.insert(END, "📁 Loading pre-trained Naive Bayes model...\n")
        clf = joblib.load('model/naive_bayes_model.pkl')
        predict = clf.predict(X_test)
        calculateMetrics("Naive Bayes", predict, y_test)
    else:
        text.insert(END, "🧠 Training new Naive Bayes model...\n")
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'model/naive_bayes_model.pkl')
        predict = clf.predict(X_test)
        calculateMetrics("Naive Bayes", predict, y_test)
    
    # Update the current model for prediction
    model_used = "Naive Bayes"

def runKNN():
    text.delete('1.0', END)
    text.insert(END, "🔍 Running K-Nearest Neighbors Classifier...\n\n")
    global X_train, X_test, y_train, y_test, clf, model_used
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    if os.path.exists('model/KNeighborsClassifier.pkl'):
        text.insert(END, "📁 Loading pre-trained KNN model...\n")
        clf = joblib.load('model/KNeighborsClassifier.pkl')
        predict = clf.predict(X_test)
        calculateMetrics("K-Nearest Neighbors", predict, y_test)
    else:
        text.insert(END, "🧠 Training new KNN model...\n")
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'model/KNeighborsClassifier.pkl') 
        predict = clf.predict(X_test)
        calculateMetrics("K-Nearest Neighbors", predict, y_test)
    
    # Update the current model for prediction
    model_used = "K-Nearest Neighbors"
    
def runRFC():
    text.delete('1.0', END)
    text.insert(END, "🔍 Running Random Forest Classifier...\n\n")
    global X_train, X_test, y_train, y_test, clf, model_used
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    if os.path.exists('model/RandomForestClassifier.pkl'):
        text.insert(END, "📁 Loading pre-trained Random Forest model...\n")
        clf = joblib.load('model/RandomForestClassifier.pkl')
        predict = clf.predict(X_test)
        calculateMetrics("Random Forest", predict, y_test)
    else:
        text.insert(END, "🧠 Training new Random Forest model...\n")
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'model/RandomForestClassifier.pkl') 
        predict = clf.predict(X_test)
        calculateMetrics("Random Forest", predict, y_test)
    
    # Update the current model for prediction
    model_used = "Random Forest"

def graph():
    text.delete('1.0', END)
    text.insert(END, "📊 Generating Performance Comparison Chart...\n\n")
    
    # Define model files and names
    model_files = {
        "Logistic Regression": "model/LogisticRegression.pkl",
        "Naive Bayes": "model/naive_bayes_model.pkl",
        "K-Nearest Neighbors": "model/KNeighborsClassifier.pkl",
        "Random Forest": "model/RandomForestClassifier.pkl"
    }
    
    # Check if models exist
    available_models = []
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            available_models.append(model_name)
    
    if not available_models:
        text.insert(END, "⚠️ Error: No trained models found!\n\n")
        text.insert(END, "Please train at least one model before comparing performance metrics.\n")
        text.insert(END, "Use the training buttons (Logistic Regression, Naive Bayes, KNN, or Random Forest) to train models.")
        return
    
    text.insert(END, f"📋 Found {len(available_models)} trained models: {', '.join(available_models)}\n\n")
    
    # Arrays to store metrics from each model
    algorithms = []
    accuracy_values = []
    precision_values = []
    recall_values = []
    fscore_values = []
    
    # For each available model, load it and calculate metrics
    for model_name in available_models:
        model_path = model_files[model_name]
        
        try:
            # Load the model
            clf = joblib.load(model_path)
            
            # Use the model to predict on test data
            predict = clf.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, predict) * 100
            prec = precision_score(y_test, predict, average='macro') * 100
            rec = recall_score(y_test, predict, average='macro') * 100
            f1 = f1_score(y_test, predict, average='macro') * 100
            
            # Store results
            algorithms.append(model_name)
            accuracy_values.append(acc)
            precision_values.append(prec)
            recall_values.append(rec)
            fscore_values.append(f1)
            
            text.insert(END, f"✅ Calculated metrics for {model_name}\n")
            
        except Exception as e:
            text.insert(END, f"⚠️ Error calculating metrics for {model_name}: {str(e)}\n")
    
    if not algorithms:
        text.insert(END, "⚠️ Error: Could not calculate metrics for any model!\n")
        return
    
    # Create arrays for plotting
    metrics_values = []
    
    # Add all models to the visualization
    for i, model in enumerate(algorithms):
        metrics_values.append([accuracy_values[i], precision_values[i], recall_values[i], fscore_values[i]])
    
    # Create DataFrame for visualization
    df_plot = pd.DataFrame({
        'Algorithm': np.repeat(algorithms, 4),
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'] * len(algorithms),
        'Value': np.concatenate([values for values in metrics_values])
    })
    
    # Plot with enhanced styling
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x='Algorithm', y='Value', hue='Metric', data=df_plot, palette='viridis')
    
    # Add a dark background with grid
    chart.set_facecolor('#121212')
    chart.figure.set_facecolor('#121212')
    chart.set_xlabel('Algorithms', fontsize=14, color='white')
    chart.set_ylabel('Performance (%)', fontsize=14, color='white')
    chart.set_title('Model Performance Comparison', fontsize=18, fontweight='bold', color='white')
    chart.tick_params(colors='white')
    
    # Add value labels on top of bars
    for p in chart.patches:
        chart.annotate(f'{p.get_height():.1f}%', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='bottom', fontsize=9, color='white')
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Metrics', fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.show()
    
    text.insert(END, "✅ Performance comparison complete!\n")
    text.insert(END, "📈 Summary of Results:\n\n")
    
    # Print summary table for all models
    for i, alg in enumerate(algorithms):
        text.insert(END, f"🚀 {alg}:\n")
        text.insert(END, f"  Accuracy:  {accuracy_values[i]:.2f}%\n")
        text.insert(END, f"  Precision: {precision_values[i]:.2f}%\n")
        text.insert(END, f"  Recall:    {recall_values[i]:.2f}%\n")
        text.insert(END, f"  F1 Score:  {fscore_values[i]:.2f}%\n\n")    


# def predictStarType():
#     """
#     Function to predict star types on test data using all trained models and compare with original type
#     """
#     text.delete('1.0', END)
#     text.insert(END, "🌟 Star Type Prediction on Test Data with Model Comparison\n\n")
    
#     # Check if models have been trained
#     model_files = {
#         "Logistic Regression": "model/LogisticRegression.pkl",
#         "Naive Bayes": "model/naive_bayes_model.pkl",
#         "K-Nearest Neighbors": "model/KNeighborsClassifier.pkl",
#         "Random Forest": "model/RandomForestClassifier.pkl"
#     }
    
#     # Check which models are available
#     available_models = {}
#     for model_name, model_path in model_files.items():
#         if os.path.exists(model_path):
#             available_models[model_name] = joblib.load(model_path)
    
#     if not available_models:
#         text.insert(END, "⚠️ Error: No trained models found!\n\n")
#         text.insert(END, "Please train at least one model first (Logistic Regression, Naive Bayes, KNN, or Random Forest).")
#         return
    
#     text.insert(END, f"📋 Found {len(available_models)} trained models: {', '.join(available_models.keys())}\n\n")
    
#     try:
#         # Get test data file
#         filename = filedialog.askopenfilename(initialdir="Dataset", title="Select Test Data File")
#         if not filename:
#             text.insert(END, "⚠️ No file selected. Operation cancelled.\n")
#             return
            
#         text.insert(END, f"📁 Loading test data from: {filename}\n\n")
#         test_data = pd.read_csv(filename)
        
#         # Check if Type column exists
#         if 'Type' not in test_data.columns:
#             text.insert(END, "⚠️ Error: Test data does not contain 'Type' column for comparison.\n")
#             return
        
#         # Store original types
#         original_types = test_data['Type'].copy()
        
#         # Create a copy of test data without the Type column for predictions
#         test_X = test_data.drop('Type', axis=1)
#         test_display = test_X.copy()  # For display purposes
        
#         # Process the test data similar to training data
#         global scaler, label_encoders
        
#         for col in test_X.select_dtypes(include=['object']).columns:
#             if col in label_encoders:
#                 test_X[col] = test_X[col].map(lambda s: label_encoders[col].transform([s])[0] if s in label_encoders[col].classes_ else -1)
        
#         # Apply scaling
#         test_scaled = scaler.transform(test_X)
        
#         # Dictionary mapping indices to star types
#         mydict = {
#             0: "Red Dwarf",
#             1: "Brown Dwarf",
#             2: "White Dwarf",
#             3: "Main Sequence",
#             4: "Super Giants",
#             5: "Hyper Giants"
#         }
        
#         # Make predictions with each model
#         predictions = {}
#         for model_name, model in available_models.items():
#             predictions[model_name] = model.predict(test_scaled)
        
#         # Display comparison tables for each test data point
#         for i in range(len(test_data)):
#             text.insert(END, f"\n✨ Test Star #{i + 1}:\n")
            
#             # Display star properties
#             text.insert(END, "Properties:\n")
#             properties = []
#             for k, v in test_display.iloc[i].items():
#                 properties.append(f"{k}: {v}")
            
#             # Display properties in a more readable format (3 properties per line)
#             for j in range(0, len(properties), 3):
#                 text.insert(END, f"   {', '.join(properties[j:j+3])}\n")
            
#             # Create comparison table
#             text.insert(END, "\nModel Predictions vs Actual Type:\n")
#             text.insert(END, "┌" + "─" * 70 + "┐\n")
#             text.insert(END, f"│ {'Model':<25} │ {'Predicted Type':<38} │\n")
#             text.insert(END, "├" + "─" * 70 + "┤\n")
            
#             # Show actual type
#             actual_type = original_types.iloc[i]
#             text.insert(END, f"│ {'ACTUAL':<25} │ {actual_type:<38} │\n")
#             text.insert(END, "├" + "─" * 70 + "┤\n")
            
#             # Show predictions from each model
#             correct_predictions = 0
#             total_predictions = 0
            
#             for model_name in available_models:
#                 predicted_idx = predictions[model_name][i]
#                 predicted_type = mydict.get(predicted_idx, "Unknown")
#                 total_predictions += 1
                
#                 # Check if prediction matches actual
#                 if predicted_type == actual_type:
#                     correct_predictions += 1
#                     result_marker = "✓"  # Correct prediction
#                 else:
#                     result_marker = "✗"  # Wrong prediction
                
#                 text.insert(END, f"│ {model_name:<25} │ {predicted_type:<36} {result_marker} │\n")
            
#             text.insert(END, "└" + "─" * 70 + "┘\n")
            
#             # Calculate accuracy for this test case
#             accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
#             text.insert(END, f"Accuracy for this star: {accuracy:.1f}% ({correct_predictions}/{total_predictions} models correct)\n")
            
#             # Add a separator between test cases
#             text.insert(END, "\n" + "─" * 75 + "\n")
        
#         # Calculate overall accuracy for each model
#         text.insert(END, "\n🔍 Overall Model Performance:\n\n")
#         text.insert(END, "┌" + "─" * 70 + "┐\n")
#         text.insert(END, f"│ {'Model':<25} │ {'Accuracy':<10} │ {'Correct/Total':<25} │\n")
#         text.insert(END, "├" + "─" * 70 + "┤\n")
        
#         for model_name, model_preds in predictions.items():
#             # Convert numeric predictions to star types
#             predicted_types = [mydict.get(pred, "Unknown") for pred in model_preds]
            
#             # Calculate accuracy
#             correct = sum(pred == actual for pred, actual in zip(predicted_types, original_types))
#             total = len(original_types)
#             accuracy = (correct / total) * 100 if total > 0 else 0
            
#             text.insert(END, f"│ {model_name:<25} │ {accuracy:.1f}% │ {correct}/{total} stars correctly classified │\n")
        
#         text.insert(END, "└" + "─" * 70 + "┘\n")
        
#         text.insert(END, f"\n✅ Prediction and comparison complete!\n")
        
#     except Exception as e:
#         text.insert(END, f"⚠️ Error during prediction: {str(e)}\n")
#         import traceback
#         text.insert(END, traceback.format_exc())

# def predictStarType():
#     """
#     Function to predict star types on new test data using the current model
#     """
#     text.delete('1.0', END)
#     text.insert(END, "🌟 Star Type Prediction on Test Data\n\n")
    
#     global clf, scaler, label_encoders, model_used
    
#     # Check if a model has been trained
#     if not model_used:
#         text.insert(END, "⚠️ Error: Please train a model first!\n")
#         text.insert(END, "Use one of the training buttons (Logistic Regression, Naive Bayes, KNN, or Random Forest) to train a model.")
#         return
    
#     try:
#         # Get test data file
#         filename = filedialog.askopenfilename(initialdir="Dataset", title="Select Test Data File")
#         if not filename:
#             text.insert(END, "⚠️ No file selected. Operation cancelled.\n")
#             return
            
#         text.insert(END, f"📁 Loading test data from: {filename}\n\n")
#         test = pd.read_csv(filename)
#         test_display = test.copy()  # Keep an unmodified copy for display
        
#         # Process the test data similarly to training data
#         for col in test.select_dtypes(include=['object']).columns:
#             if col in label_encoders:
#                 test[col] = test[col].map(lambda s: label_encoders[col].transform([s])[0] if s in label_encoders[col].classes_ else -1)
        
#         # Apply scaling
#         test_scaled = scaler.transform(test)
        
#         # Make predictions
#         predictions = clf.predict(test_scaled)
        
#         # Dictionary mapping indices to star types
#         mydict = {
#             0: "Red Dwarf",
#             1: "Brown Dwarf",
#             2: "White Dwarf",
#             3: "Main Sequence",
#             4: "Super Giants",
#             5: "Hyper Giants"
#         }
        
#         # Display predictions with cosmic styling
#         text.insert(END, f"🔮 Prediction Results Using {model_used} Model:\n\n")
#         text.insert(END, "┌" + "─" * 70 + "┐\n")
#         text.insert(END, "│" + " " * 25 + "STAR PREDICTIONS" + " " * 29 + "│\n")
#         text.insert(END, "└" + "─" * 70 + "┘\n\n")
        
#         # Iterate through each row of the dataset
#         for index, row in test_display.iterrows():
#             # Get the prediction for the current row
#             prediction = predictions[index]
#             predicted_type = mydict.get(prediction, "Unknown")
            
#             # Display with cosmic styling
#             text.insert(END, f"✨ Star #{index + 1}:\n")
            
#             # Format properties nicely
#             properties = []
#             for k, v in row.items():
#                 properties.append(f"{k}: {v}")
            
#             # Display properties in a more readable format (limit to 3 properties per line)
#             for i in range(0, len(properties), 3):
#                 text.insert(END, f"   {', '.join(properties[i:i+3])}\n")
            
#             text.insert(END, f"   🌠 Predicted Type: {predicted_type}\n\n")
            
#         text.insert(END, f"✅ Prediction complete using {model_used} model!\n")
        
#     except Exception as e:
#         text.insert(END, f"⚠️ Error during prediction: {str(e)}\n")
#         import traceback
#         text.insert(END, traceback.format_exc())


def predict_custom_star():
    """
    Function to predict star type based on user input values
    """
    global clf, scaler, model_used, label_encoders
    
    text.delete('1.0', END)
    text.insert(END, "🌟 Custom Star Type Prediction\n\n")
    
    # Check if a model has been trained
    if not model_used:
        text.insert(END, "⚠️ Error: Please train a model first!\n")
        text.insert(END, "Use one of the training buttons (Logistic Regression, Naive Bayes, KNN, or Random Forest) to train a model.")
        return
        
    # Create a new window for input
    input_window = Toplevel(main)
    input_window.title("Star Properties Input")
    input_window.configure(bg=SPACE_BG)
    input_window.geometry("600x700")
    
    # Create a frame for inputs
    input_frame = Frame(input_window, bg=PANEL_BG, bd=2, relief=RIDGE)
    input_frame.pack(padx=20, pady=20, fill=BOTH, expand=True)
    
    # Title
    title_label = Label(input_frame, text="🔭 Enter Star Properties", bg=PANEL_BG, fg=STAR_GOLD,
                       font=('Helvetica', 16, 'bold'), pady=10)
    title_label.pack(fill=X)
    
    # Scrollable canvas for many fields
    canvas = Canvas(input_frame, bg=PANEL_BG, highlightthickness=0)
    scrollbar = Scrollbar(input_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas, bg=PANEL_BG)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    scrollbar.pack(side="right", fill="y")
    
    # Dictionary to store input fields
    input_fields = {}
    
    # Function to create a labeled entry field
    def create_field(parent, label_text, default=""):
        field_frame = Frame(parent, bg=PANEL_BG, pady=5)
        field_frame.pack(fill=X, padx=10)
        
        label = Label(field_frame, text=label_text, bg=PANEL_BG, fg=TEXT_COLOR,
                     font=('Helvetica', 12), width=15, anchor="w")
        label.pack(side=LEFT, padx=5)
        
        entry = Entry(field_frame, font=('Helvetica', 12), bg="#0A0F20", fg=TEXT_COLOR,
                     insertbackground=TEXT_COLOR, width=25)
        entry.pack(side=LEFT, padx=5, fill=X, expand=True)
        entry.insert(0, default)
        
        return entry
    
    # Get the feature names from the model's expected input
    if hasattr(df, 'columns'):
        # Exclude the target column 'Type'
        feature_names = [col for col in df.columns if col != 'Type']
        
        # Create input fields for each feature
        for feature in feature_names:
            # Get default values from the dataset median
            if feature in df.columns and df[feature].dtype in [np.float64, np.int64]:
                default = str(df[feature].median())
            else:
                default = ""
                
            input_fields[feature] = create_field(scrollable_frame, f"{feature}:", default)
    else:
        text.insert(END, "⚠️ Error: Please load and process a dataset first!\n")
        input_window.destroy()
        return
    
    # Function to handle prediction
    def make_prediction():
        try:
            # Collect input values
            input_data = {}
            for feature, entry in input_fields.items():
                value = entry.get().strip()
                
                # Convert to appropriate type
                if feature in df.columns:
                    if df[feature].dtype == np.float64:
                        input_data[feature] = float(value)
                    elif df[feature].dtype == np.int64:
                        input_data[feature] = int(value)
                    else:  # For categorical features
                        input_data[feature] = value
            
            # Create a DataFrame from the input
            input_df = pd.DataFrame([input_data])
            
            # Apply the same preprocessing steps as the training data
            for col in input_df.select_dtypes(include=['object']).columns:
                if col in label_encoders:
                    input_df[col] = input_df[col].map(lambda s: label_encoders[col].transform([s])[0] if s in label_encoders[col].classes_ else -1)
            
            # Apply scaling
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = clf.predict(input_scaled)[0]
            
            # Map prediction to star type
            mydict = {
                0: "Red Dwarf",
                1: "Brown Dwarf",
                2: "White Dwarf",
                3: "Main Sequence",
                4: "Super Giants",
                5: "Hyper Giants"
            }
            predicted_type = mydict.get(prediction, "Unknown")
            
            # Close input window
            input_window.destroy()
            
            # Display result with cosmic styling
            text.delete('1.0', END)
            text.insert(END, "🔮 Custom Star Prediction Result:\n\n")
            text.insert(END, "┌" + "─" * 70 + "┐\n")
            text.insert(END, "│" + " " * 25 + "STAR PREDICTION" + " " * 30 + "│\n")
            text.insert(END, "└" + "─" * 70 + "┘\n\n")
            
            # Display input properties
            text.insert(END, "✨ Star Properties:\n")
            properties = []
            for k, v in input_data.items():
                properties.append(f"{k}: {v}")
            
            # Display properties in a more readable format (3 properties per line)
            for i in range(0, len(properties), 3):
                text.insert(END, f"   {', '.join(properties[i:i+3])}\n")
            
            # Display prediction with cosmic styling
            text.insert(END, f"\n🌠 Predicted Star Type: {predicted_type}\n\n")
            
            # Add the star type description
            descriptions = {
                "Red Dwarf": "Small, cool, and long-lived stars with mass approximately 0.075-0.5 solar masses and surface temperatures under 4,000K.",
                "Brown Dwarf": "Failed stars with insufficient mass for sustained nuclear fusion, typically below 0.075 solar masses.",
                "White Dwarf": "Remnant cores of stars that have shed their outer layers, typically Earth-sized but with a mass comparable to the Sun.",
                "Main Sequence": "Stars like our Sun that are fusing hydrogen into helium in their cores, representing about 90% of all stars.",
                "Super Giants": "Enormous stars with high luminosity, mass 10-70 solar masses, and near the end of their life cycle.",
                "Hyper Giants": "Extremely massive and luminous stars, over 100 solar masses, which are quite rare in the universe."
            }
            
            text.insert(END, f"📝 About {predicted_type} stars:\n")
            text.insert(END, f"   {descriptions.get(predicted_type, 'No description available.')}\n\n")
            
            text.insert(END, f"✅ Prediction complete using {model_used} model!\n")
            
        except Exception as e:
            text.insert(END, f"⚠️ Error during prediction: {str(e)}\n")
            import traceback
            text.insert(END, traceback.format_exc())
    
    # Buttons frame
    buttons_frame = Frame(input_window, bg=PANEL_BG, pady=10)
    buttons_frame.pack(fill=X, padx=20, pady=10)
    
    # Predict button
    predict_btn = Button(buttons_frame, text="Predict Star Type", command=make_prediction,
                       bg=COSMIC_BLUE, fg=TEXT_COLOR, font=('Helvetica', 12, 'bold'),
                       activebackground=NEBULA_PURPLE, activeforeground='white',
                       relief=RAISED, bd=2, padx=10, pady=8)
    predict_btn.pack(side=LEFT, padx=10)
    
    # Cancel button
    cancel_btn = Button(buttons_frame, text="Cancel", command=input_window.destroy,
                      bg=MARS_RED, fg=TEXT_COLOR, font=('Helvetica', 12, 'bold'),
                      activebackground=NEBULA_PURPLE, activeforeground='white',
                      relief=RAISED, bd=2, padx=10, pady=8)
    cancel_btn.pack(side=RIGHT, padx=10)


# Create buttons with cosmic styling
create_cosmic_button(button_frame, "🔍 Upload Dataset", upload)
create_cosmic_button(button_frame, "🧪 Preprocess Data", processDataset)
create_cosmic_button(button_frame, "📊 Exploratory Analysis", perform_eda)
create_cosmic_button(button_frame, "🔄 Train Logistic Regression", runLRC)
create_cosmic_button(button_frame, "🔄 Train Naive Bayes", runNB)
create_cosmic_button(button_frame, "🔄 Train KNN", runKNN)
create_cosmic_button(button_frame, "🔄 Train Random Forest", runRFC)
create_cosmic_button(button_frame, "📈 Evaluate Performance", graph)
# create_cosmic_button(button_frame, "📈 Test Star Type", predictStarType)
create_cosmic_button(button_frame, "🔭 Predict Custom Star", predict_custom_star)

# Add a separator
separator = Frame(button_frame, height=2, bg=STAR_GOLD)
separator.pack(fill=X, padx=20, pady=10)

# Add the prediction button at the bottom with new function
# create_cosmic_button(button_frame, "🌟 Predict Star Types", predictStarType)

# Footer with version info
footer_frame = Frame(main, bg=PANEL_BG)
footer_frame.place(x=0, y=screen_height-30, width=screen_width, height=30)
footer_label = Label(footer_frame, 
                    text="🚀 NASA Star Classification System v1.0 | © 2025 Stellar Research Group", 
                    bg=PANEL_BG, fg=TEXT_COLOR)
footer_label.pack(fill=BOTH, expand=True)

# Welcome message
text.insert(END, """
✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨
                       WELCOME TO THE NASA STAR CLASSIFICATION SYSTEM
✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨

This application uses advanced machine learning techniques to classify stars
into six categories based on their properties:

🔴 Red Dwarf       - Small, cool, and long-lived stars
🟤 Brown Dwarf      - Failed stars with insufficient mass for nuclear fusion
⚪ White Dwarf      - Remnant cores of stars that have shed their outer layers
🟡 Main Sequence    - Stars like our Sun, burning hydrogen into helium
🟠 Super Giants     - Enormous stars with high luminosity nearing end of life
🔵 Hyper Giants     - Extremely massive and luminous stars

To begin:
1. Click "Upload Dataset" to load your stellar data
2. Process the data with "Preprocess Data"
3. Explore your data with "Exploratory Analysis"
4. Train and evaluate different models
5. Make predictions on new star data

Let's explore the cosmos together!
""")

# Start the application
main.mainloop()