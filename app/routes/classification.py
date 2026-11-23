from tabfpn.model_loading import load_fitted_tabpfn_model
import os
import json
model_path="app\models\trained_reg.tabpfn_fit"
with open("saved_models/feature_columns.json") as f:
    features = json.load(f)

model = load_fitted_tabpfn_model(model_path)
