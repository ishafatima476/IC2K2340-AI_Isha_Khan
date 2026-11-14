"""
AI Mini Project: Student Performance Prediction
Uses a Decision Tree Classifier on a synthetic educational dataset.
The prediction is for learning/demo purposes only.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "student_performance.csv"
MODEL_PATH = ROOT / "models" / "student_performance_model.pkl"


def load_data():
    """Read the CSV dataset and check that it is not empty."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    if data.empty:
        raise ValueError("The dataset is empty.")
    return data


def train_model(data):
    """Train and evaluate a Decision Tree classification model."""
    feature_names = [
        "study_hours_per_day",
        "attendance_percent",
        "previous_exam_score",
        "assignment_score",
    ]
    X = data[feature_names]
    y = data["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    print("\\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(classifier, MODEL_PATH)
    return classifier


def predict_new_student(classifier):
    """Ask for student details and display the predicted category."""
    try:
        study_hours = float(input("\\nStudy hours per day (0-24): "))
        attendance = float(input("Attendance percentage (0-100): "))
        previous_score = float(input("Previous exam score (0-100): "))
        assignment_score = float(input("Assignment score (0-100): "))

        if not 0 <= study_hours <= 24:
            raise ValueError("Study hours must be between 0 and 24.")
        for value in (attendance, previous_score, assignment_score):
            if not 0 <= value <= 100:
                raise ValueError("Attendance and scores must be between 0 and 100.")

        student = pd.DataFrame([{
            "study_hours_per_day": study_hours,
            "attendance_percent": attendance,
            "previous_exam_score": previous_score,
            "assignment_score": assignment_score,
        }])
        print("Predicted result:", classifier.predict(student)[0])
        print("Note: This is an educational demo, not a real academic decision.")
    except ValueError as error:
        print("Invalid input:", error)


def main():
    try:
        data = load_data()
        print("Dataset loaded successfully.")
        print(data.head())
        classifier = train_model(data)
        predict_new_student(classifier)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print("Project error:", error)


if __name__ == "__main__":
    main()
