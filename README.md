# NASA Star Classification System 🌟

## About the Project
The NASA Star Classification System is a sophisticated machine learning application that helps classify stars into different categories based on their physical properties. This project utilizes various ML algorithms to accurately categorize stars into six distinct types: Red Dwarf, Brown Dwarf, White Dwarf, Main Sequence, Super Giants, and Hyper Giants.

## Features 🚀

### Data Management
- **Dataset Upload**: Easily import stellar data through a user-friendly interface
- **Data Preprocessing**: Automated data cleaning and preparation for machine learning
- **Exploratory Data Analysis (EDA)**: Comprehensive visualization tools for understanding star distributions and relationships

### Machine Learning Models
The system implements four powerful classification algorithms:
- **Logistic Regression**: For linear classification boundaries
- **Naive Bayes**: Probabilistic classification using Bayes' theorem
- **K-Nearest Neighbors (KNN)**: Classification based on closest training examples
- **Random Forest**: Ensemble learning method for improved accuracy

### Analysis Tools
- **Performance Metrics**: Detailed accuracy, precision, recall, and F1-score measurements
- **Visual Analytics**: Interactive graphs and charts for model comparison
- **Custom Predictions**: Ability to predict star types using custom input parameters

### User Interface
- **Modern Space Theme**: Dark mode interface with cosmic styling
- **Interactive Elements**: User-friendly buttons and input fields
- **Real-time Results**: Immediate feedback on predictions and analysis

## Setup Instructions 🛠️

### Prerequisites
- Python 3.x
- Required Python packages:
  - tkinter
  - pandas
  - numpy
  - matplotlib
  - plotly
  - seaborn
  - scikit-learn
  - imbalanced-learn
  - pillow
  - joblib

### Installation
1. Clone the repository
2. Install required packages:
```bash
pip install pandas numpy matplotlib plotly seaborn scikit-learn imbalanced-learn pillow joblib
```
3. Place your dataset in the `Dataset` folder
4. Run `spacecenter.py`

## Usage Guide 📖

1. **Data Loading**
   - Click "Upload Dataset" to select your star data file
   - The system accepts CSV files with star properties

2. **Data Preprocessing**
   - Use "Preprocess Data" to prepare your data for analysis
   - The system will automatically handle:
     - Data normalization
     - Feature encoding
     - Train-test splitting

3. **Exploratory Analysis**
   - Click "Exploratory Analysis" to view:
     - Feature distributions
     - Correlation matrices
     - Star type distributions

4. **Model Training**
   - Train different models using dedicated buttons
   - Models are automatically saved for future use
   - Compare performance metrics across models

5. **Making Predictions**
   - Use "Predict Custom Star" to classify new stars
   - Input star properties manually
   - Get instant predictions with explanations

## Star Types Classification 🌠

The system classifies stars into six categories:

1. **Red Dwarf**
   - Small, cool stars
   - Mass: 0.075-0.5 solar masses
   - Temperature: < 4,000K

2. **Brown Dwarf**
   - Failed stars
   - Mass: < 0.075 solar masses
   - Insufficient mass for nuclear fusion

3. **White Dwarf**
   - Stellar remnants
   - Earth-sized but sun-like mass
   - Final evolutionary state

4. **Main Sequence**
   - Like our Sun
   - Stable hydrogen fusion
   - Most common star type

5. **Super Giants**
   - Massive stars
   - Mass: 10-70 solar masses
   - Near end of life cycle

6. **Hyper Giants**
   - Extremely massive
   - Mass: > 100 solar masses
   - Rare in universe

## Project Structure 📁

```
project/
│
├── Dataset/                # Data directory
│   ├── Stars.csv          # Training dataset
│   └── test1.csv          # Test dataset
│
├── model/                 # Saved ML models
│   ├── KNeighborsClassifier.pkl
│   ├── LogisticRegression.pkl
│   ├── naive_bayes_model.pkl
│   └── RandomForestClassifier.pkl
│
├── spacecenter.py        # Main application file
├── background.jpg        # UI background image
└── README.md            # Documentation
```

## Contributing 🤝

Feel free to fork this project and submit pull requests with improvements. Some areas for potential enhancement:
- Additional machine learning algorithms
- More visualization options
- Extended star classification categories
- Performance optimizations

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- NASA for stellar classification standards
- The scientific community for star classification research
- Open source machine learning libraries

## Contact 📧

For questions and support, please open an issue in the repository.

---
Built with ❤️ for astronomical research and education.
