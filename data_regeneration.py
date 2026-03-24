import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=2000, shift_type='none', random_state=42):
    """
    Generates synthetic data with nonlinear interactions.
    y = X1 * X2 + X3^2 + sin(X4) + noise
    
    shift_type: 
      - 'none': In-distribution (ID) data.
      - 'covariate': Shifts the mean of features, altering the data distribution.
      - 'variance': Increases the variance of the features.
    """
    np.random.seed(random_state)
    
    if shift_type == 'none':
        mean = 0.0
        scale = 1.0
    elif shift_type == 'covariate':
        mean = 2.5  # Mean shift
        scale = 1.0
    elif shift_type == 'variance':
        mean = 0.0
        scale = 3.0 # High variance shift
    else:
        raise ValueError("Invalid shift_type")
        
    X1 = np.random.normal(mean, scale, n_samples)
    X2 = np.random.normal(mean, scale, n_samples)
    X3 = np.random.normal(mean, scale, n_samples)
    X4 = np.random.normal(mean, scale, n_samples)
    X5 = np.random.normal(mean, scale, n_samples) # purely noise feature
    
    # Target generation (complex nonlinear function)
    y = X1 * X2 + X3**2 + np.sin(X4) + np.random.normal(0, 0.5, n_samples)
    
    df = pd.DataFrame({
        'Feature_1': X1,
        'Feature_2': X2,
        'Feature_3': X3,
        'Feature_4': X4,
        'Noise_Feature': X5
    })
    
    return df, y
