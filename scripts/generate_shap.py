
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import pickle
import shap

def plot_shap_summary():
    print("Generating SHAP Summary Plot...")
    try:
        # Load Model
        with open('../results/sota_quick.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # Load Data (Small sample for SHAP)
        # We need the feature matrix used for prediction
        # Since we didn't save X_val in quick run, we quickly regenerate it
        # ideally we should have saved it. 
        # For now, let's trying to load the model and if we can't easily get X, we might skip
        # BUT, we can use the main.py logic to get a small df
        
        from src.data_loader import load_data
        from src.feature_engineering import engineer_features
        from main import run_experiment
        
        # This is a bit heavy, let's just make a dummy or reload
        df = load_data('dataset', sample_size=1000, is_train=True)
        df = engineer_features(df, use_magic=True)
        exclude = ['isFraud', 'TransactionID', 'TransactionDT', 'uid', 'uid_encoded']
        model_features = [c for c in df.columns if c not in exclude]
        X = df[model_features]
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # LightGBM binary classification: shap_values is list, index 1 is positive class
        # Depending on version it might be array
        if isinstance(shap_values, list):
            vals = shap_values[1]
        else:
            vals = shap_values
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(vals, X, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig('../results/shap_summary.png', dpi=300)
        print("Saved results/shap_summary.png")
        
    except Exception as e:
        print(f"Failed to plot SHAP: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_shap_summary()
