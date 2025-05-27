import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection   import SelectFromModel, RFE
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

def plot_model_analysis(model, X_test, y_test, feature_order, model_name):
    """
    Unified function for model visualization including:
    - Feature importance
    - Confusion matrix
    """
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Feature Importance
    plt.subplot(1, 2, 1)
    if hasattr(model, 'feature_importances_'):  # For RF and XGBoost
        importance = np.nan_to_num(model.feature_importances_)
    elif hasattr(model, 'coef_'):  # For Logistic Regression
        importance = np.nan_to_num(np.abs(model.coef_[0]))
    else:
        raise AttributeError("Model doesn't have feature importance attributes")
    
    importance_df = pd.DataFrame({
        'feature': feature_order,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title(f'Feature Importance - {model_name}')
    plt.xlabel('Importance Score')
    
    # Plot 2: Confusion Matrix
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results, Y_test):
    """
    Function to plot ROC curves for all models
    """
    plt.figure(figsize=(10, 6))
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(Y_test, result['Probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Load and pre-process data
data = pd.read_csv("C:\\Users\\yushi\\OneDrive\\Desktop\\learning data\\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

X = data.drop('Diabetes_binary', axis=1)
Y = data['Diabetes_binary']

numerical_features = ['BMI', 'MentHlth', 'PhysHlth']
binary_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
                  'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                  'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 
                  'DiffWalk', 'Sex']
ordinal_features = ['GenHlth', 'Education', 'Income', 'Age']

scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

for feature in ordinal_features:
    X[feature] = X[feature] / X[feature].max()

# First split into train+validation and test
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y,
                                                 test_size=0.2,
                                                 random_state=42,
                                                 stratify=Y)

# Then split train+validation into separate sets
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp,
                                                 test_size=0.25,  # 0.25 x 0.8 = 0.2
                                                 random_state=42,
                                                 stratify=Y_temp)

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10]
}

param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    ),
    'XGBoost': GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid_xgb,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
}

def add_baseline_model(X_train, X_test, Y_train, Y_test, results):
    # Create stratified random baseline (most appropriate for balanced dataset)
    baseline = DummyClassifier(strategy='stratified', random_state=42)
    
    # Train the baseline model
    baseline.fit(X_train, Y_train)
    
    # Get predictions
    y_pred = baseline.predict(X_test)
    y_proba = baseline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(Y_test, y_pred)
    cv_scores = cross_val_score(baseline, X_train, Y_train, cv=5)
    
    # Store results in the same format as other models
    results['Baseline'] = {
        'Accuracy': accuracy,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Predictions': y_pred,
        'Probabilities': y_proba
    }
    
    print("\nBaseline Model Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cross-validation scores: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print("\nClassification Report:")
    print(classification_report(Y_test, y_pred))
    
    return results

# Dictionary to store results
results = {}

# Train and evaluate each model
# Train and evaluate each model
for name, model in models.items():
    print(f"\n{'-'*50}\nAnalyzing {name}:")
    
    # Train the model
    model.fit(X_train, Y_train)
    
    # If it's a GridSearchCV object, get the best model and its parameters
    if isinstance(model, GridSearchCV):
        print(f"Best parameters for {name}:")
        print(model.best_params_)
        y_pred = model.best_estimator_.predict(X_test)
        y_proba = model.best_estimator_.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(Y_test, y_pred)
    cv_scores = cross_val_score(model, X, Y, cv=5)
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Predictions': y_pred,
        'Probabilities': y_proba
    }
    
    # Print detailed results
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Cross-validation scores: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print("\nClassification Report:")
    print(classification_report(Y_test, y_pred))

results = add_baseline_model(X_train, X_test, Y_train, Y_test, results)

# Get models for analysis
rf_model = models['Random Forest'].best_estimator_ if isinstance(models['Random Forest'], GridSearchCV) else models['Random Forest']
lr_model = models['Logistic Regression']
xgb_model = models['XGBoost'].best_estimator_ if isinstance(models['XGBoost'], GridSearchCV) else models['XGBoost']



# Model Stacking
def create_stacking_model():
    estimators = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
        ('lr', LogisticRegression(random_state=42))
    ]
    
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    return stack

# Define the desired order of features
feature_order = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
                'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

# Plot analysis for each model
for name, model in [('Logistic Regression', lr_model), 
                   ('Random Forest', rf_model), 
                   ('XGBoost', xgb_model)]:
    print(f"\n{name} Analysis:")
    plot_model_analysis(model, X_test, Y_test, feature_order, name)

# ROC Curves
plt.figure(figsize=(10, 6))
for name, result in results.items():
    fpr, tpr, _ = roc_curve(Y_test, result['Probabilities'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Print final comparison table
comparison_df = pd.DataFrame({
    'Test Accuracy': [results[model]['Accuracy'] for model in models],
    'CV Mean': [results[model]['CV Mean'] for model in models],
    'CV Std': [results[model]['CV Std'] for model in models]
}, index=models.keys())

print("\nFinal Model Comparison:")
print(comparison_df.round(4))


def analyze_shap_values(model, X, model_name):
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model) if model_name in ['Random Forest', 'XGBoost'] else shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f'SHAP Feature Importance - {model_name}')
    plt.tight_layout()
    plt.show()
    
    # Individual feature impact
    plt.figure(figsize=(10, 8))
    shap.dependence_plot("BMI", shap_values, X, show=False)
    plt.title(f'SHAP Dependence Plot for BMI - {model_name}')
    plt.tight_layout()
    plt.show()



# Create and train stacking model
print("\nTraining stacking classifier...")
stack = create_stacking_model()
stack.fit(X_train, Y_train)
stack_pred = stack.predict(X_test)
stack_accuracy = accuracy_score(Y_test, stack_pred)
print(f"\nStacking Classifier Accuracy: {stack_accuracy:.4f}")
print("\nStacking Classifier Classification Report:")
print(classification_report(Y_test, stack_pred))
