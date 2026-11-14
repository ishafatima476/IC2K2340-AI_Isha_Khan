import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Create dataset
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 2, 5],
    'Attendance': [40, 45, 50, 60, 65, 70, 80, 90, 55, 75],
    'Result': ['Fail', 'Fail', 'Fail', 'Fail', 'Pass',
               'Pass', 'Pass', 'Pass', 'Fail', 'Pass']
}

df = pd.DataFrame(data)

# Separate input and output
X = df[['Study_Hours', 'Attendance']]
y = df['Result']

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict for a new student
new_student = pd.DataFrame({
    'Study_Hours': [6],
    'Attendance': [80]
})

prediction = model.predict(new_student)
print("Student Result:", prediction[0])

# Visualize Decision Tree
plt.figure(figsize=(12, 7))
plot_tree(
    model,
    feature_names=['Study_Hours', 'Attendance'],
    class_names=model.classes_,
    filled=True
)
plt.show()
