from pycaret.classification import setup, compare_models, create_model, finalize_model, plot_model

model_holder = {}
feature_importance_holder = {}

for target in model_targets:
    # Step 1: Drop model_targets column except the current target
    data = data_train.drop(columns=[col for col in model_targets if col != target])

    # Set up the PyCaret environment
    clf_setup = setup(data=data, target=target, session_id=123)

    # Compare and choose the best performing models
    best_models = compare_models()

    # Get the best model for the current target
    best_model = create_model(best_models, fold=10)

    # Step 2: Train the best model for the current target
    clf_pycaret = finalize_model(best_model)

    # Save the best model in the model_holder dictionary
    model_holder[target] = clf_pycaret

    # Get feature importance for the best model
    feature_importance = plot_model(best_model, plot='feature', save=True)
    
    # Store feature importance in the feature_importance_holder dictionary
    feature_importance_holder[target] = feature_importance
