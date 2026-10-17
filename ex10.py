
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Sample Dataset
data = {
    'text': [
        'Win a free lottery now',
        'Meeting scheduled at 10am tomorrow',
        'Claim your free prize immediately',
        'Please review the attached report',
        'Congratulations you won cash prize',
        'Let us discuss the project update',
        'Get cheap loans instantly approved',
        'Lunch with team at 1pm',
        'Free entry in weekly contest',
        'Reminder submit your assignment'
    ],
    'label': [
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham'
    ]
}

df = pd.DataFrame(data)

# Step 2: Features and Labels
X = df['text']
y = df['label']

# Step 3: Convert text into numeric vectors
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)

# Step 4: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_vect, y, test_size=0.3, random_state=42,
    stratify=y
)

# Step 5: Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Predict test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# Step 8: Test a new email
new_email = [
    "Congratulations! You have won a free iPhone, claim now"
]

new_vect = vectorizer.transform(new_email)
prediction = model.predict(new_vect)

print("\nNew Email Prediction:", prediction[0])
