# Feature-Analysis
# Mechanistic Feature Analysis — Tree-Based Models

This repository investigates how internal decision structures in tree-based models (Random Forest, XGBoost) degrade under data distribution shift. 

It highlights a core reliability problem for machine learning models running at scale: model explanations (like SHAP) can become highly inconsistent and brittle when the model is forced to extrapolate on out-of-distribution (OOD) data.

## Project Structure

- `src/data_generation.py`: Generates a synthetic regression dataset featuring complex interactions ($X_1 \cdot X_2$) and non-linearities ($X_3^2$, $\sin(X_4)$). Includes logic to create both In-Distribution (ID) and Out-Of-Distribution (OOD) splits.
- `src/model_training.py`: Handles training of tree-based ensembles (Random Forest, XGBoost).
- `src/perturbation_analysis.py`: Contains the core logic for calculating SHAP (SHapley Additive exPlanations) values and measuring their consistency under local neighborhood perturbation. 
- `main.py`: Orchestrates the entire experiment, computing performance degradation ($R^2$, MSE) and generating visualization plots.
- `results/`: Directory containing generated plots and metrics.

## Key Concepts Investigated

1. **Performance Degradation under Shift**: Models that achieve $>0.9 R^2$ on In-Distribution data often fail catastrophically when shifted (e.g., Covariate shift where the mean of continuous features changes).
2. **Attribution Consistency**: Model explanations are supposed to help us trust models. However, we demonstrate that when models extrapolate on OOD data, their decision paths become chaotic. As a result, small perturbations (e.g., adding slight noise to an input vector) cause drastically different SHAP explanation profiles. 
3. **Implications**: The brittleness of explanations means that we cannot blindly trust local attribution methods for heavily shifted data.

## How to Run

1. Make sure you have the dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the main experiment script:
   ```bash
   python main.py
   ```
3. Inspect the outputs in the `results/` folder to compare differences between In-Distribution, Covariate-Shifted, and Variance-Shifted test sets.
