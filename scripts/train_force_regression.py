import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
def train_correlated_model(X, y, test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create a pipeline that:
    # 1. Scales the data (crucial for Ridge and PCA)
    # 2. Applies PCA (compresses 4 features down to 2 most important independent components)
    # 3. Applies Ridge Regression
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=2), 
        Ridge(alpha=1.0)
    )
        
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    metrics = {
        'Test MSE': mean_squared_error(y_test, predictions),
        'Test R2': r2_score(y_test, predictions)
    }
    
    return model, metrics

with open('data/force_exp.pkl', 'rb') as f:
    data_all = pickle.load(f)
    initial_cable_force = data_all['initial_cable_force']
    cable_force_list = data_all['cable_force_list']
    contact_force_list = data_all['contact_force_list']

train_force_list = []
for i in range(len(cable_force_list)):
    fl_input = []
    for j in range(4):
        fl_input.append(cable_force_list[i][j] - initial_cable_force[j])
    train_force_list.append(fl_input)

model, metrics = train_correlated_model(np.array(train_force_list), contact_force_list)
print("Model Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

model_pickleFile = 'models/force_regression_model.pkl'
with open(model_pickleFile, 'wb') as f:
    pickle.dump(model, f)

