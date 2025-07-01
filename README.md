# Resume Classification Model

This project is a machine learning pipeline to classify resumes into job categories using text preprocessing, feature engineering, and supervised learning models.

## 🔍 Problem Statement
Given a dataset of resumes and their associated job categories, the goal is to build a classification model that predicts the most suitable category for a given resume.

## 🧰 Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy

## 🧪 Models Tried
- Support Vector Machine (SVM)
- Logistic Regression

Both models achieved approximately **70% weighted F1-score**, which is considered a solid performance for this dataset given its imbalance and variability.

## 🛠️ Preprocessing Steps
- Lowercasing text
- Removing special characters and numbers
- Feature extraction: word count, char count, keyword counts
- TF-IDF vectorization for resume text
- Feature selection using SelectKBest and chi²
- Standard scaling for numerical features

## 📊 Final Model
The final model is trained using a full pipeline that includes preprocessing and classification using Logistic Regression. It’s saved and ready for inference.

## 📝 Results
- Final F1-score: ~0.70 (weighted)
- Cross-validation F1-score: ~0.71
- test set F1-score: 0.68

## 🚀 Future Improvements
- Try deep learning approaches (BERT or transformers)
- Handle rare categories more effectively


---

**Author:** [omar]  
**Date:** July 2025
