from pathlib import Path
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

DATASET = Path(__file__).resolve().parent.parent / "dataset" / "queries.csv"

FALLBACK_DATA = [
    ("What are the symptoms of diabetes?", "medical"),
    ("What is hypertension?", "medical"),
    ("What causes anemia?", "medical"),
    ("What is pneumonia?", "medical"),
    ("How is high blood pressure diagnosed?", "medical"),
    ("What are side effects of antibiotics?", "medical"),
    ("What is an ECG used for?", "medical"),
    ("What is an MRI scan?", "medical"),
    ("What is a blood glucose test?", "medical"),
    ("What are symptoms of influenza?", "medical"),
    ("What is a heart attack?", "medical"),
    ("What is insulin?", "medical"),
    ("How is anemia treated?", "medical"),
    ("What are signs of dehydration?", "medical"),
    ("What is appendicitis?", "medical"),
    ("What is chemotherapy?", "medical"),
    ("What are symptoms of migraine?", "medical"),
    ("What is a drug interaction?", "medical"),
    ("What is a vaccine?", "medical"),
    ("What is a biopsy?", "medical"),
    ("What is asthma?", "medical"),
    ("What is sepsis?", "medical"),
    ("What is a bacterial infection?", "medical"),
    ("What is a viral infection?", "medical"),
    ("What is a clinical trial?", "medical"),
    ("What is a pathology test?", "medical"),
    ("What does an ECG measure?", "medical"),
    ("What are symptoms of tuberculosis?", "medical"),
    ("What is diabetes mellitus?", "medical"),
    ("How is pneumonia diagnosed?", "medical"),
    ("What is hypertension treatment?", "medical"),
    ("What are symptoms of gastritis?", "medical"),
    ("What is an antibiotic?", "medical"),
    ("What is a blood pressure reading?", "medical"),
    ("What is emergency medicine?", "medical"),
    ("What is preventive healthcare?", "medical"),
    ("What is mental health?", "medical"),
    ("What is depression?", "medical"),
    ("What are symptoms of anxiety?", "medical"),
    ("What is an X-ray?", "medical"),
    ("What is a laboratory test?", "medical"),
    ("What is the thyroid gland?", "medical"),
    ("What does a neurologist treat?", "medical"),
    ("What is kidney disease?", "medical"),
    ("What is a surgical procedure?", "medical"),
    ("What is a medication dose?", "medical"),
    ("What is a clinical diagnosis?", "medical"),

    ("What is Python?", "non-medical"),
    ("Write a Python program to calculate factorial.", "non-medical"),
    ("Who won the cricket match?", "non-medical"),
    ("Tell me a joke.", "non-medical"),
    ("What is the capital of India?", "non-medical"),
    ("Write a Java program.", "non-medical"),
    ("How can I learn web development?", "non-medical"),
    ("What is HTML?", "non-medical"),
    ("Explain machine learning.", "non-medical"),
    ("What is the weather today?", "non-medical"),
    ("Recommend a movie.", "non-medical"),
    ("What is the stock market?", "non-medical"),
    ("Who is the Prime Minister of India?", "non-medical"),
    ("Plan a trip to Mumbai.", "non-medical"),
    ("How do I cook pasta?", "non-medical"),
    ("Write a poem about friendship.", "non-medical"),
    ("What is the latest cricket score?", "non-medical"),
    ("How do I edit a photo?", "non-medical"),
    ("Explain quantum computing.", "non-medical"),
    ("What is SQL?", "non-medical"),
    ("How do I create a website?", "non-medical"),
    ("Tell me a funny story.", "non-medical"),
    ("What is the history of India?", "non-medical"),
    ("Which phone should I buy?", "non-medical"),
    ("How do I learn JavaScript?", "non-medical"),
    ("What is a database?", "non-medical"),
    ("What is a neural network?", "non-medical"),
    ("Write an email to my teacher.", "non-medical"),
    ("Who won the football game?", "non-medical"),
    ("What is the population of India?", "non-medical"),
    ("Give me a study timetable.", "non-medical"),
    ("What is artificial intelligence?", "non-medical"),
    ("How do I make a presentation?", "non-medical"),
    ("Explain blockchain.", "non-medical"),
    ("What is C++?", "non-medical"),
    ("Write a program to sort an array.", "non-medical"),
    ("What is democracy?", "non-medical"),
    ("Recommend a restaurant.", "non-medical"),
    ("How do I travel to Delhi?", "non-medical"),
    ("Tell me about Bollywood movies.", "non-medical"),
    ("What is Excel?", "non-medical"),
    ("How do I make a YouTube channel?", "non-medical"),
    ("What is a computer virus?", "non-medical"),
    ("How do I install Python?", "non-medical"),
    ("Write a short story.", "non-medical"),
    ("What is the best laptop for students?", "non-medical"),
    ("Explain cloud computing.", "non-medical"),
    ("What is cryptocurrency?", "non-medical"),
    ("How do I create a logo?", "non-medical"),
    ("Solve this math equation.", "non-medical"),
]

class MedicalQueryClassifier:
    def __init__(self, threshold=0.70):
        self.threshold = threshold
        rows = []
        if DATASET.exists():
            with DATASET.open("r", encoding="utf-8") as f:
                rows = [(r["question"], r["label"]) for r in csv.DictReader(f)]
        if len(rows) < 20:
            rows = FALLBACK_DATA

        texts = [x[0] for x in rows]
        labels = [x[1] for x in rows]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        X = self.vectorizer.fit_transform(texts)

        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="lbfgs",
            max_iter=1200,
            random_state=42
        )
        self.model.fit(X, labels)

    def predict(self, text):
        X = self.vectorizer.transform([text])
        probabilities = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        i = int(probabilities.argmax())
        label = classes[i]
        confidence = float(probabilities[i])

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "is_medical": label == "medical" and confidence >= self.threshold
        }
