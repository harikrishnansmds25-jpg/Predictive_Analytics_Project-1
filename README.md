# Predictive_Analytics_Project-1
Mental Health Status Classification from Social Media Text
🧠 Mental Health Status Classification from Social Media Posts
> **Academic Research Project · Machine Learning · NLP**  
> *For Research and Educational Purposes Only — Not Intended for Clinical Diagnosis*
---
📌 Overview
This project builds an automated 4-class text classification system to detect early mental health signals in anonymised social media posts. Online platforms host millions of posts containing subtle linguistic signals of psychological distress — manual moderation cannot scale. This system addresses that gap using a classical NLP + ML pipeline that is fast, interpretable, and CPU-trainable.
Category	Description
😞 Depression	Persistent sadness, hopelessness, loss of interest or energy
😰 Anxiety	Excessive worry, panic, fear or nervousness disrupting daily life
😨 PTSD	Post-traumatic patterns: flashbacks, hypervigilance, avoidance
🙂 Normal	Everyday posts with no significant mental health distress indicators
---
📁 Repository Contents
File	Description
`index.html`	Deployed interactive web application — includes project overview, methodology, live demo, results, and ethics pages
`MentalHealthClassification_Final.pptx`	Full academic presentation (24 slides) covering problem statement, methodology, results, screenshots, and conclusions
---
📊 Dataset
Source: Kaggle — Sentiment Analysis for Mental Health (anonymised Reddit & Twitter posts)
Total Samples: 5,957 anonymised social media posts
Balance: ~1,190 posts per class (balanced)
Features: `text`, `title`, `target`
Preprocessing Steps
Remove duplicates and posts under 10 characters
Filter extreme length outliers (top 1%)
Standardise label column to canonical values
Lowercase, strip URLs, HTML tags & punctuation
Remove NLTK stopwords; apply WordNet lemmatisation
Preserve negations via bigrams (`ngram_range=(1,2)`)
---
⚙️ Methodology
Pipeline
```
Raw Text → Clean & Preprocess → TF-IDF Vectorise → SVM Classify → Evaluate → Ethical Review
```
TF-IDF Vectorisation
Converts text into high-dimensional numerical features using term frequency weighted by inverse document frequency.
Parameter	Value	Rationale
`max_features`	8,000	Most informative unigrams + bigrams
`ngram_range`	(1, 2)	Captures word-pairs like "not happy"
`min_df`	3	Filters noise (rare terms)
`max_df`	0.85	Filters domain-generic words
`sublinear_tf`	True	Log scaling prevents long posts dominating
Support Vector Machine
Model: `LinearSVC` wrapped in `CalibratedClassifierCV` (5-fold CV) for probability scores
Strategy: One-vs-rest multiclass
Class weighting: `balanced` — automatically corrects for any imbalance
Normal Guard: If no mental health class confidence exceeds `0.45`, post is classified as Normal — key fix for over-prediction
Why TF-IDF + SVM over BERT?
Fully interpretable feature weights
Trains in < 30 seconds on CPU — no GPU required
Reproducible and academically transparent
~5–8% accuracy trade-off is acceptable for an early-support research tool
---
📈 Results
Category	Precision	Recall	Specificity	F1-Score
Depression	0.82	0.80	0.91	0.81
Anxiety	0.78	0.77	0.92	0.77
PTSD	0.75	0.76	0.94	0.75
Normal ★	0.89	0.91	0.93	0.90
Macro Avg	0.81	0.81	0.93	0.81
Key highlights:
✅ Primary design target MET: 93% Specificity on Normal class (TNR ≥ 0.90)
✅ Cross-validation F1-Macro: `0.80 ± 0.02` (5-fold) — stable generalisation
✅ Training time: < 30 seconds on CPU only
✅ Vocabulary: 8,000 features
---
🌐 Deployed Web Application
The project is deployed as an interactive academic website (`index.html`) featuring:
Overview Page — 4 classification categories with colour-coded cards and problem statement
Methodology Page — Pipeline diagram and detailed technical steps
Results & Metrics Page — Full classification report with KPI cards
Live Demo — Lexical Approximation Engine for real-time classification
Ethics & Limitations Page — Responsible AI considerations
Demo Examples:
> *"Had such a fun day hiking with friends today!"* → ✔ **NORMAL** — 100% Confidence
> *"Every loud sound makes me jump. I can't sleep without nightmares."* → ✔ **PTSD** — 100% Confidence
---
⚠️ Ethics & Limitations
Concern	Details
Not a Diagnostic Tool	Classifies text patterns, not individuals. Must NEVER be used for clinical, legal, or insurance decisions.
Data Privacy	All posts are fully anonymised. No names, usernames, or location data retained.
Bias & Fairness	Training data is skewed toward English-speaking Western populations. May underperform on non-standard dialects.
False Negative Risk	At 75–80% recall, ~20–25% of at-risk posts may be missed. Human review remains essential.
Intended Use	Research trend analysis, content moderation support, early-warning population signals only.
Prohibited Use	Individual profiling or surveillance.
---
🔭 Future Work
BERT Fine-tuning — Fine-tune BERT / Mental-RoBERTa for 5–8% accuracy gain
Multilingual Support — Extend to non-English datasets
Temporal Retraining — Automated pipeline with concept-drift detection
Clinical Validation — Collaborate with mental health professionals
Explainability — SHAP/LIME integration for token-level explanations
Real-time API — REST API for social platform moderation pipelines
---
🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![HTML5](https://img.shields.io/badge/HTML5-Web-red?logo=html5)
Language: Python 3.11
ML: scikit-learn (`LinearSVC`, `CalibratedClassifierCV`, `TfidfVectorizer`)
NLP: NLTK (`stopwords`, `WordNetLemmatizer`)
Frontend: HTML5, CSS3, JavaScript (vanilla)
---
📚 References
Coppersmith, G., Dredze, M., & Harman, C. (2014). Quantifying mental health signals in Twitter. ACL Workshop on CL & Clinical Psychology.
Gkotsis, G., et al. (2017). Characterisation of mental health conditions in social media using informed deep learning. Scientific Reports, 7(1), 45141.
Losada, D. E., & Crestani, F. (2016). A test collection for research on depression and language use. CLEF.
Vapnik, V. N. (1995). The Nature of Statistical Learning Theory. Springer.
Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.
Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly.
Kaggle Dataset — Sentiment Analysis for Mental Health (anonymised Reddit/Twitter posts).
---
> **Academic Research Project · Department of Computer Science**  
> *For Research and Educational Purposes Only · Not Intended for Clinical Diagnosis*
