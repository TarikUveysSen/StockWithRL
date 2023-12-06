from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from pycaret.classification import *

# Assuming 'train_pipe1' is your training data and 'Y_train' is your target variable

# Set up the PyCaret environment
clf_setup = setup(data=train_pipe1, target=Y_train.columns.tolist(), session_id=123, silent=True, multi_label=True)

# Compare and choose the best performing models
best_model = compare_models()

model_holder = {}

# Train the best models for each target using OneVsRestClassifier
for target in Y_train.columns:
    # Wrap the PyCaret best model with OneVsRestClassifier
    clf_pycaret = OneVsRestClassifier(LogisticRegression())  # You can replace LogisticRegression() with the best model from PyCaret
    clf_pycaret.fit(get_config("X_train"), get_config("y_train")[target])
    model_holder[target] = clf_pycaret
