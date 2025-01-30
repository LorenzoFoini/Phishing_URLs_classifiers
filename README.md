# URLs phishing detection using Machine Learning
![](phishing_URL.jpg)

## Description
This project was developed as part of the **Artificial Intelligence for Security** course, where I explored different approaches applicable in cybersecurity. The goal is to classify URLs as legitimate or phishing using machine learning algorithms.

## Project Structure
- **Dataset**: The dataset used can be found at `Dataset\Phishing_URL_Dataset.csv` in this repository. It contains **235795 URLs**, with a label indicating whether the URL is phishing or legitimate.
- **Data Exploration**: An exploratory data analysis was performed to identify relevant insights. Various features, their values, and their distribution relative to the target were visualized.
- **Data Preprocessing**: The confusion matrix was analyzed to obtain a general idea of the dataset. Subsequently, the features with the highest correlation were manually checked to prevent overfitting and maximimse the performances fo the different classifiers.
- **Supervised Learning**: Several machine learning models were developed:
  - **Linear Classifiers**: Logistic Regression (accuracy **99.644%**), Gaussian Naive Bayes (accuracy **97.363%**), Support Vector Machine (accuracy **99.706%**).
  - **Non-Linear Classifiers**: Decision Tree (accuracy **99.754%**), K-Nearest Neighbors (accuracy **98.647%**), Deep Neural Network (accuracy **99.713%**).
  - **Ensemble Classifiers**: XGBoost (accuracy **99.942%**), Random Forest (accuracy **99.884%**).
  - A comparison between various models was also conducted using recall, precision, ROC AUC, and calibration metrics.
- **Unsupervised Learning**: Clustering algorithms were implemented:
  - K-Means, DBSCAN, and Hierarchical Clustering.
  - Performance was evaluated using homogeneity, completeness, v-measure, silhouette score, and Calinsky-Harabasz score.
- **Anomaly Detection**: Several algorithms were used to detect anomalies:
  - Elliptic Envelope, One-Class SVM, Isolation Forest, and Local Outlier Factor.
  - Finally, the decision boundaries of the various algorithms were visualized.

## Requirements
To run the project, install the following libraries:
```bash
!pip install numpy pandas scikit-learn matplotlib seaborn missingno lime
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/your-username/repository-name.git
```
2. Navigate to the project folder:
```bash
cd repository-name
```
3. Run the notebook:
```bash
jupyter notebook Project_Casaleggi_Fatigati_Foini_Martinelli.ipynb
```

## Disclaimer
This project is intended for educational and research purposes only. The application of the algorithm outside of educational contexts is not recommended, and security is not guaranteed. The authors assume no responsibility for any consequences arising from the use of this code in real-world scenarios.

## License
This project is for educational purposes only and follows the course requirements of the **Artificial Intelligence for Security** course.  
All rights are reserved by the author, and the code is provided exclusively for academic use.
