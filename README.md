<p align="center">
  <img src="https://img.shields.io/badge/TripAdvisor%20Review%20Rating%20Prediction-NLP%20Project-green?style=for-the-badge&logo=python&logoColor=white" />
</p>

<p align="center">
  <b>Natural Language Processing • Text Classification • TF-IDF • Bag of Words • ML Models</b>
</p>

# TripAdvisor Review Rating Prediction Using NLP & Machine Learning

This project focuses on predicting hotel review ratings based solely on the textual content of customer reviews from TripAdvisor.
Using Natural Language Processing (NLP) and machine learning techniques, the goal is to automatically classify how satisfied a customer is by analyzing their written feedback.

⭐ Project Aim
--

To analyze TripAdvisor hotel reviews using NLP techniques.

To convert raw text into numerical features using TF-IDF and CountVectorizer.

To build ML models that predict customer ratings (1–5 stars).

To automate the review classification process and assist businesses in understanding customer sentiment.

📂 Dataset Information
--

Rows: 20,491

Columns: 2

Review: Text written by customers

Rating: Numerical value (1–5)

Each row represents a hotel review given by a customer on TripAdvisor.
The dataset is suitable for text classification, sentiment analysis, and rating prediction.

🧾 Feature Information
--
| Feature | Description |
|---------|-------------|
| Review  | Customer-written review text describing their hotel experience, feedback, opinions, and sentiments. |
| Rating  | Star rating (1–5) assigned by the user, representing the level of customer satisfaction. |


🛠 Technologies & Libraries Used
--

Python

NumPy, Pandas

Matplotlib, Seaborn

NLTK (stopwords)

Scikit-learn:

TfidfVectorizer

CountVectorizer

Logistic Regression

Linear SVM

Naive Bayes

Train/Test Split

Metrics (accuracy, confusion matrix, F1-score)

📊 Exploratory Data Analysis
--

Review length distribution

Rating count distribution

Review length vs rating

WordCloud of frequently occurring keywords

Visualizations help understand writing patterns and sentiment distribution across ratings.

⭐ Conclusion
--

* Logistic Regression produced the best performance among all models.

* TF-IDF vectorization resulted in higher accuracy compared to CountVectorizer.

* Review text contains strong patterns that help predict customer satisfaction.

* NLP is effective for automating large-scale review analysis.

🧠 Models Trained & Performance
--
| Model                         | Accuracy |
|-------------------------------|----------|
| Naive Bayes (TF-IDF)          | 61.0%    |
| Linear SVM                    | 62.5%    |
| Logistic Regression           | 65.0%    |
| Naive Bayes (CountVectorizer) | 54.5%    |

**Conclusion:** Logistic Regression outperformed all other models, achieving the highest accuracy of 65%.

🚀 Future Enhancements
--

Implement deep learning models (LSTM/BERT).

Add sentiment polarity (+ve/–ve) detection.

Hyperparameter tuning for improved accuracy.

Deploy the model using Flask/Streamlit.

📁 Project File
--

* This repository includes the full project script : Trip_advisor.ipynb

[Trip_advisor.ipynb](https://github.com/user-attachments/files/23879749/Trip_advisor.ipynb)
