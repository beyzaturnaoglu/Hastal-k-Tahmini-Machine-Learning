import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import messagebox

# Veri setini yükleme
file_path = 'Disease_symptom_and_patient_profile_dataset.xlsx'
data = pd.read_excel(file_path)

# Veri setinden gereksiz veya boş sütunları çıkar
data_cleaned = data.drop(columns=['Unnamed: 10'], errors='ignore').dropna()

# Kategorik verileri sayısal formata dönüştürme
label_encoders = {}
for column in data_cleaned.columns:
    data_cleaned[column] = data_cleaned[column].astype(str)
    le = LabelEncoder()
    data_cleaned[column] = le.fit_transform(data_cleaned[column])
    label_encoders[column] = le

# Özellikler ve hedef değişkeni ayırma
X = data_cleaned.drop('Disease', axis=1)
y = data_cleaned['Disease']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Rastgele orman sınıflandırıcısı modelini kullanma
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Tkinter ile GUI oluşturma
root = tk.Tk()
root.title("Hastalık Tahmini")

# Kullanıcı girişleri için etiketler ve giriş kutuları oluşturma
labels = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable']
entries = {}
for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entries[label] = entry

# Tahmin fonksiyonu
def predict_disease():
    new_data = {label: entries[label].get() for label in labels}
    for column in new_data:
        if column in label_encoders:
            new_data[column] = label_encoders[column].transform([new_data[column]])[0]
    new_df = pd.DataFrame([new_data])
    prediction = model.predict(new_df)
    predicted_label = label_encoders['Disease'].classes_[prediction[0]]
    messagebox.showinfo("Tahmin Sonucu", f"Tahmin edilen hastalık: {predicted_label}")

# Tahmin butonu
predict_button = tk.Button(root, text="Tahmin Et", command=predict_disease)
predict_button.grid(row=len(labels), column=0, columnspan=2)

# GUI başlatma
root.mainloop()
