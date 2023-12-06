for target, model in model_holder.items():
    # Extract feature importance as a list
    feature_importance_list = model.get_feature_importance()

    # Print or use the feature importance list as needed
    print(f"Feature Importance for {target}: {feature_importance_list}")

    # Plot feature importance
    plot_model(model, plot='feature', save=True, title=f'Feature Importance - {target}')
