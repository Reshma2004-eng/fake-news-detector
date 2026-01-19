import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

# 1. Load Data
print("⏳ Loading Datasets...")
try:
    df_fake = pd.read_csv('data/Fake.csv')
    df_true = pd.read_csv('data/True.csv')
except FileNotFoundError:
    print("❌ Error: 'data' folder lo True.csv mariyu Fake.csv ledu. Please check.")
    exit()

# 2. Add Labels (0 = Fake, 1 = Real)
df_fake["class"] = 0
df_true["class"] = 1

# 3. Combine Data
# Manam Title + Text rendu combine chesthe accuracy baga vastundi
df_fake['text'] = df_fake['title'] + " " + df_fake['text']
df_true['text'] = df_true['title'] + " " + df_true['text']

# Manual Testing kosam last 10 rows remove chestunnam (optional)
df_manual_testing = pd.concat([df_fake.tail(10), df_true.tail(10)], axis=0)
df_fake = df_fake.iloc[:-10]
df_true = df_true.iloc[:-10]

# Final Merge
df = pd.concat([df_fake, df_true], axis=0)
df = df.sample(frac=1).reset_index(drop=True) # Shuffle data

print(f"✅ Data Loaded. Total News Articles: {df.shape[0]}")

# 4. Text Cleaning Function (Reliability kosam)
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

print("🧹 Cleaning Text (Idi konchem time padtundi)...")
df["text"] = df["text"].apply(wordopt)

# 5. Define X and Y
x = df["text"]
y = df["class"]

# 6. Split Data (80% Training, 20% Testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# 7. Convert Text to Numbers (Vectorization)
print("🧮 Converting text to vectors...")
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# 8. Train Model (PassiveAggressiveClassifier)
print("🚀 Training Model...")
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(xv_train, y_train)

# 9. Check Accuracy
pred_pac = model.predict(xv_test)
score = accuracy_score(y_test, pred_pac)
print(f"🎉 Model Training Complete!")
print(f"🎯 Accuracy Score: {score*100:.2f}%")

# 10. Save Model
if not os.path.exists('models'):
    os.makedirs('models')
    
joblib.dump(model, 'models/model.pkl')
joblib.dump(vectorization, 'models/vectorizer.pkl')
print("💾 Model saved in 'models/' folder.")
