# 🛡️ TruthGuard: AI-Powered Fake News Detector

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Framework](https://img.shields.io/badge/Frontend-Streamlit-red)

## 📌 Project Overview
**TruthGuard** is a Machine Learning-based web application designed to combat the spread of misinformation. It analyzes news articles or headlines and classifies them as **Real** or **Fake** using Natural Language Processing (NLP) techniques.

This project was developed as part of an **AI/ML Internship** to demonstrate the application of supervised learning in text classification.

## 🚀 Features
* **Instant Analysis:** Users can paste news text/headlines to get immediate feedback.
* **High Accuracy:** Built using the **Passive Aggressive Classifier**, achieving ~99% accuracy on the test dataset.
* **Interactive UI:** Developed using **Streamlit** for a clean, user-friendly experience.
* **Confidence Metrics:** Visual indicators (Green for Real, Red for Fake) to guide the user.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Joblib
* **NLP Techniques:** TF-IDF Vectorization, Stop-word removal, Regex cleaning
* **Web Framework:** Streamlit

## 📂 Project Structure
```text
FakeNewsDetector/
│
├── data/                   # Dataset folder
│   ├── True.csv            # Real news articles
│   └── Fake.csv            # Fake news articles
│
├── models/                 # Saved ML models (Generated after training)
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── train_model.py          # Script to train and save the model
├── app.py                  # Streamlit frontend application
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation

data set is too large to upload,and .pkl files not need for streamlit cloud connection.
⚙️ Installation & Setup
Prerequisites
Python installed on your system.

Dataset files (True.csv and Fake.csv) placed inside the data/ folder.

Step 1: Clone or Download
Download the project folder to your local machine.

Step 2: Install Dependencies
Open your terminal/command prompt, navigate to the project folder, and run:

Bash

pip install -r requirements.txt
Step 3: Train the Model
Before running the app, you need to train the AI model. Run:

Bash

python train_model.py
This will generate the model.pkl and vectorizer.pkl files in the models/ directory.

Step 4: Run the Application
Launch the web interface using Streamlit:

Bash

streamlit run app.py
📊 Model Performance
Algorithm: Passive Aggressive Classifier

Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency)

Accuracy: ~99.4% (Based on ISOT Dataset split)

🧪 How to Test
Run the app.

Copy a news paragraph (Real or Fake).

Paste it into the text box.

Click "Analyze News".

🔮 Future Scope
Adding URL scraping feature to analyze news directly from links.

Implementing Deep Learning models (BERT/LSTM) for better context understanding.

Multilingual support (Hindi, Telugu, etc.).

Developed by: Sakalabattula Reshma
 Internship Domain: Artificial Intelligence & Machine Learning
