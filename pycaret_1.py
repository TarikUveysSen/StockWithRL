from pycaret.classification import *

# Assuming 'train_pipe1' is your training data and 'Y_train' is your target variable

# Set up the PyCaret environment
clf_setup = setup(data=train_pipe1, target=Y_train.columns.tolist(), session_id=123, silent=True, multi_label=True)

# Compare and choose the best performing models
best_models = compare_models()

model_holder = {}

# Train the best models for each target
for target in Y_train.columns:
    best_model = create_model(best_models, fold=10)
    clf_pycaret = finalize_model(best_model)
    model_holder[target] = clf_pycaret
