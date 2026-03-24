import numpy as np
import shap

def compute_attribution_consistency(model, X, perturbation_scale=0.1, n_perturbations=10):
    """
    Measures the consistency of SHAP attributions under local neighborhood perturbation.
    For each sample x, we generate perturbed samples x' = x + noise.
    Consistency is measured as the average L2 distance between SHAP(x) and SHAP(x').
    Higher values mean the explanations are more inconsistent/brittle.
    """
    explainer = shap.TreeExplainer(model)
    
    # Baseline SHAP values
    shap_baseline = explainer.shap_values(X)
    if isinstance(shap_baseline, list): # RF might return list in older shap versions
        shap_baseline = shap_baseline[0]
        
    n_samples, n_features = X.shape
    accumulated_distances = np.zeros(n_samples)
    
    for _ in range(n_perturbations):
        # Generate slightly perturbed inputs (local neighborhood)
        noise = np.random.normal(0, perturbation_scale, (n_samples, n_features))
        X_perturbed = X + noise
        
        # Calculate SHAP values for perturbed inputs
        shap_perturbed = explainer.shap_values(X_perturbed)
        if isinstance(shap_perturbed, list):
            shap_perturbed = shap_perturbed[0]
            
        # Distance (L2 norm of explanation difference vectors)
        distances = np.linalg.norm(shap_baseline - shap_perturbed, axis=1)
        accumulated_distances += distances
        
    avg_distances = accumulated_distances / n_perturbations
    return avg_distances
