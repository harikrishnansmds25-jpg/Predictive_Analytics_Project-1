Predictive_Analytics_Project-1
Mental Health Status Classification from Social Media Posts

Academic Research Project · Machine Learning · NLP
For Research and Educational Purposes Only — Not Intended for Clinical Diagnosis

📌 Overview

This project presents an automated 4-class text classification system designed to detect early mental health indicators from anonymized social media posts.

Millions of users share thoughts, emotions, and experiences online. Many posts contain subtle linguistic markers of psychological distress, making manual monitoring impractical at scale.

This project addresses that challenge using a classical NLP + Machine Learning pipeline that is:

Fast and CPU-efficient
Interpretable
Academically reproducible
Suitable for research-scale deployment
🎯 Classification Categories
Category	Description
😞 Depression	Persistent sadness, hopelessness, loss of interest or low energy
😰 Anxiety	Excessive worry, panic, fear, or nervousness disrupting daily life
😨 PTSD	Flashbacks, hypervigilance, avoidance, trauma-related distress
🙂 Normal	Everyday posts with no major mental health distress indicators
🌐 Live Deployment

The project is deployed as an interactive HTML-based web application.

🔗 Live App

Launch Application

Deployment Type

✅ Static HTML Deployment using GitHub Pages

The deployed website includes:

Project Overview
Methodology
Results Dashboard
Interactive Demo
Ethics & Limitations
📁 Repository Contents
File	Description
index.html	HTML deployed interactive web application
MentalHealthClassification_Final.pptx	Final academic project presentation
README.md	Project documentation
Model notebooks/scripts	Data preprocessing, training, evaluation
📊 Dataset

Source: Kaggle — Sentiment Analysis for Mental Health
(Anonymized Reddit & Twitter posts)

Dataset Statistics
Total Samples: 5,957
Classes: 4
Balanced Distribution: ~1,190 per class
Features
Post text
Title
Target label
🧹 Data Preprocessing

The dataset underwent extensive preprocessing:

Duplicate removal
Filtering posts under 10 characters
Outlier length filtering
Label standardization
Lowercasing
URL & HTML removal
Punctuation stripping
Stopword removal
WordNet lemmatization
Bigram preservation for negation handling
⚙️ Methodology
NLP Pipeline

Raw Text
⬇
Clean & Preprocess
⬇
TF-IDF Vectorization
⬇
SVM Classification
⬇
Evaluation
⬇
Ethical Review

🔍 Feature Engineering
TF-IDF Vectorizer Parameters
Parameter	Value
max_features	8000
ngram_range	(1,2)
min_df	3
max_df	0.85
sublinear_tf	True

This captures:

Important keywords
Contextual word pairs
Negation patterns such as “not happy”
🤖 Model
Support Vector Machine

Classifier:
LinearSVC + CalibratedClassifierCV

Configuration
Multiclass strategy: One-vs-Rest
5-fold cross-validation
Balanced class weighting
Probability calibration
Normal Guard

A confidence threshold of 0.45 ensures that low-confidence distress predictions are safely classified as Normal, reducing overprediction.

❓ Why TF-IDF + SVM Instead of BERT?

This project prioritizes:

✅ Interpretability
✅ Faster CPU training (<30 sec)
✅ Academic transparency
✅ Lower computational requirements

While transformer models may improve accuracy by ~5–8%, classical ML remains ideal for reproducible academic research.

📈 Results
Category	Precision	Recall	Specificity	F1-Score
Depression	0.82	0.80	0.91	0.81
Anxiety	0.78	0.77	0.92	0.77
PTSD	0.75	0.76	0.94	0.75
Normal ⭐	0.89	0.91	0.93	0.90
Macro Avg	0.81	0.81	0.93	0.81
✅ Key Highlights
93% Normal Class Specificity
Cross-validation Macro F1: 0.80 ± 0.02
Training time under 30 seconds
8,000 feature vocabulary
Stable multiclass performance
💻 Web Application Features

The deployed HTML application includes:

📖 Overview Page

Project motivation and category explanations

⚙️ Methodology Page

Pipeline diagrams and model workflow

📊 Results Dashboard

Performance metrics and evaluation summaries

🧪 Live Demo

Real-time lexical approximation classifier

⚠️ Ethics Page

Responsible AI use and limitations

Example Predictions

Input:
"Had such a fun day hiking with friends today!"

Prediction:
✔ NORMAL

Input:
"Every loud sound makes me jump. I can't sleep without nightmares."

Prediction:
✔ PTSD

⚠️ Ethics & Limitations
Concern	Details
Not a Diagnostic Tool	For research only
Privacy	Fully anonymized data
Bias	English-language dataset limitations
False Negatives	Human review remains essential
Intended Use	Trend analysis & moderation support
Prohibited Use	Individual profiling or surveillance
🔭 Future Work
BERT / Mental-RoBERTa fine-tuning
Multilingual classification
Concept drift detection
Clinical validation
Explainability using SHAP/LIME
REST API deployment
🛠️ Tech Stack

Languages & Frameworks

Python 3.11
HTML5
CSS3
JavaScript

Libraries

scikit-learn
NLTK
NumPy
Pandas
📚 References
Coppersmith et al. (2014)
Gkotsis et al. (2017)
Losada & Crestani (2016)
Vapnik (1995)
Pedregosa et al. (2011)
Bird, Klein & Loper (2009)
👥 Contributors
Harikrishnan
Umaparvathy C S
Krithika S
🎓 Academic Context

Department of Computer Science
Predictive Analytics Research Project

📜 Disclaimer

This project is strictly intended for:

Academic research
Educational demonstration
NLP experimentation

It must not be used for:

Clinical diagnosis
Medical decision-making
Insurance/legal judgments
Personal surveillance
