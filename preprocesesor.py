import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Assuming data_train and data_test are your training and test DataFrames

# Create transformers
numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Select numeric and categorical features
numeric_features = data_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data_train.select_dtypes(include=['object']).columns

# Create ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)], remainder='passthrough')

# Fit and transform the data
train_pipe1 = preprocessor.fit_transform(data_train)
test_pipe1 = preprocessor.transform(data_test)

# Get the feature names for the transformed columns
numeric_feature_names = numeric_features.tolist()
categorical_feature_names = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_feature_names + categorical_feature_names

# Create DataFrames with the transformed data and column names
train_df = pd.DataFrame(train_pipe1, columns=all_feature_names)
test_df = pd.DataFrame(test_pipe1, columns=all_feature_names)
