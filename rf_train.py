import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer


df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size= 0.2, random_state= 42
)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(kernel="rbf")
}

results = {}
print("Cross-Validation Results:")

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc"
    )

    results[name] = cv_scores.mean()
    print(f"{name}: ROC-AUC = {cv_scores.mean():}")

best_model_name = max(results, key=results.get)
print("Best Model Selected:", best_model_name)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", models[best_model_name])
])


#train model
pipeline.fit(X_train, y_train)

cv_scores = cross_val_score(
    pipeline, X_train, y_train, cv = 5, scoring="roc_auc"
)

print("CV_Mean: ", cv_scores.mean())
print("CV_std: ", cv_scores.std())

param_grid = {
    "LogisticRegression": {
        "model__C": [0.01, 0.1, 1, 10]
    },
    "RandomForest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
    },
    "GradientBoosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5]
    },
    "SVM": {
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale", "auto"]
    }
}

grid = GridSearchCV(
    pipeline,
    param_grid[best_model_name],
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)


best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(" Logistic Regression pipeline saved as best_model.pkl")