from src.data_generation import generate_synthetic_data
from src.model_training import train_random_forest, train_xgboost
from src.perturbation_analysis import compute_attribution_consistency

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def run_experiment():
    os.makedirs('results', exist_ok=True)
    
    print("1. Generating In-Distribution (ID) and Shifted (OOD) Data...")
    X_id, y_id = generate_synthetic_data(n_samples=3000, shift_type='none', random_state=42)
    X_ood_cov, y_ood_cov = generate_synthetic_data(n_samples=1000, shift_type='covariate', random_state=100)
    X_ood_var, y_ood_var = generate_synthetic_data(n_samples=1000, shift_type='variance', random_state=200)
    
    # Split ID data into Train/Test
    X_train, X_test_id, y_train, y_test_id = train_test_split(X_id, y_id, test_size=0.33, random_state=42)
    
    print("2. Training Models (Random Forest and XGBoost) on ID Data...")
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    models = {'Random Forest': rf_model, 'XGBoost': xgb_model}
    datasets = {'ID Test': (X_test_id, y_test_id), 
                'OOD (Covariate Shift)': (X_ood_cov, y_ood_cov),
                'OOD (Variance Shift)': (X_ood_var, y_ood_var)}
    
    print("3. Evaluating Predictive Performance Degradation...")
    performance_records = []
    for model_name, model in models.items():
        for data_name, (X_eval, y_eval) in datasets.items():
            y_pred = model.predict(X_eval)
            r2 = r2_score(y_eval, y_pred)
            mse = mean_squared_error(y_eval, y_pred)
            performance_records.append({'Model': model_name, 'Dataset': data_name, 'R2': r2, 'MSE': mse})
            
    df_perf = pd.DataFrame(performance_records)
    print("\n--- Performance Summary ---")
    print(df_perf)
    df_perf.to_csv('results/performance_metrics.csv', index=False)
    
    print("\n4. Analyzing Attribution Consistency (SHAP) under Perturbation...")
    consistency_records = []
    
    # For speed, we evaluate on a subset of samples
    sample_size = 200
    
    for model_name, model in models.items():
        for data_name, (X_eval, y_eval) in datasets.items():
            print(f"  Measuring for {model_name} on {data_name}...")
            # Sample data
            X_sample = X_eval.sample(n=sample_size, random_state=42)
            
            # Compute attribution distances
            distances = compute_attribution_consistency(model, X_sample, perturbation_scale=0.1, n_perturbations=10)
            
            for dist in distances:
                consistency_records.append({
                    'Model': model_name,
                    'Dataset': data_name,
                    'Explanation L2 Variation': dist
                })
                
    df_consistency = pd.DataFrame(consistency_records)
    
    print("5. Plotting Results...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Predictive Performance Degradation
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_perf, x='Dataset', y='R2', hue='Model')
    plt.title("Model Performance ($R^2$) vs Distribution Shift")
    plt.ylabel("$R^2$ Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('results/performance_degradation.png')
    plt.close()
    
    # Plot 2: Explanation Brittleness (Consistency)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_consistency, x='Dataset', y='Explanation L2 Variation', hue='Model')
    plt.title("Attribution Consistency under Local Perturbation (Higher = More Brittle)")
    plt.ylabel("Mean L2 Variation in SHAP Values")
    plt.tight_layout()
    plt.savefig('results/attribution_consistency.png')
    plt.close()
    
    print("Experiment completed successfully! See 'results/' folder for outputs.")

if __name__ == "__main__":
    run_experiment()
