import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class FlightPredictionMLP(nn.Module):
    def __init__(self, input_size):
        super(FlightPredictionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, input_size)
        )
    
    def forward(self, x):
        return self.network(x)

def minimal_preprocess(filepath: str):
    """Minimal preprocessing - only handle format issues"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    initial_count = len(df)
    print(f"\nInitial entries: {initial_count}")
    
    features = ['alt', 'gs', 'heading', 'lat', 'lon', 'vertRate']
    
    # Convert to numeric, keeping all values
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"Found {nan_count} missing/invalid values in {col}")
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled missing values with median: {median_val}")
    
    # Handle altChange encoding
    df['altChange'] = df['altChange'].fillna(' ')
    df['altChange_encoded'] = df['altChange'].map({' ': 0, 'C': 1, 'D': -1}).fillna(0)
    
    print(f"\nFinal entries: {len(df)}")
    return df

def create_synthetic_anomalies(df, num_anomalies=100):
    """Generate synthetic anomalies with multiple modifications per anomaly"""
    anomalies = []
    for _ in range(num_anomalies):
        base_flight = df.sample(n=1).iloc[0].to_dict()
        anomaly = base_flight.copy()
        
        # Apply multiple modifications to each anomaly
        num_modifications = np.random.randint(1, 4)  # 1-3 modifications
        for _ in range(num_modifications):
            anomaly_type = np.random.choice([
                'sudden_altitude_change',
                'impossible_speed',
                'erratic_heading',
                'position_jump',
                'vertical_rate_anomaly',
                'inconsistent_values'
            ])
            
            if anomaly_type == 'sudden_altitude_change':
                anomaly['alt'] += np.random.choice([-15000, 15000])
                anomaly['vertRate'] = np.random.choice([-8000, 8000])
                
            elif anomaly_type == 'impossible_speed':
                anomaly['gs'] = np.random.uniform(700, 1000)
                
            elif anomaly_type == 'erratic_heading':
                anomaly['heading'] = (float(base_flight['heading']) + np.random.uniform(150, 180)) % 360
                
            elif anomaly_type == 'position_jump':
                anomaly['lat'] += np.random.uniform(-5, 5)
                anomaly['lon'] += np.random.uniform(-5, 5)
                
            elif anomaly_type == 'vertical_rate_anomaly':
                anomaly['vertRate'] = np.random.choice([-12000, 12000])
                
            elif anomaly_type == 'inconsistent_values':
                if np.random.random() > 0.5:
                    # High altitude but low ground speed
                    anomaly['alt'] = 40000
                    anomaly['gs'] = 100
                else:
                    # Low altitude but very high ground speed
                    anomaly['alt'] = 1000
                    anomaly['gs'] = 600
        
        anomalies.append(anomaly)
    
    anomaly_df = pd.DataFrame(anomalies)
    print(f"\nCreated {num_anomalies} synthetic anomalies")
    return anomaly_df

class AnomalyDetector:
    def __init__(self, model, scaler, threshold_multiplier=2.0):
        self.model = model
        self.scaler = scaler
        self.threshold = None
        self.threshold_multiplier = threshold_multiplier
    
    def fit_threshold(self, normal_data):
        """Determine threshold using normal data"""
        self.model.eval()
        with torch.no_grad():
            normal_tensor = torch.FloatTensor(normal_data)
            predictions = self.model(normal_tensor)
            reconstruction_errors = torch.mean((normal_tensor - predictions) ** 2, dim=1)
            
            # Use mean + std for threshold
            self.threshold = (torch.mean(reconstruction_errors) + 
                            self.threshold_multiplier * torch.std(reconstruction_errors))
            print(f"\nSet anomaly threshold at: {self.threshold:.6f}")
    
    def predict(self, data, return_scores=False):
        """Predict anomalies in new data"""
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data)
            predictions = self.model(data_tensor)
            reconstruction_errors = torch.mean((data_tensor - predictions) ** 2, dim=1)
            
            anomalies = reconstruction_errors > self.threshold
            
            if return_scores:
                return anomalies.numpy(), reconstruction_errors.numpy()
            return anomalies.numpy()
    
    def evaluate(self, normal_data, anomaly_data):
        """Evaluate detector performance"""
        normal_predictions, normal_scores = self.predict(normal_data, return_scores=True)
        false_positives = np.sum(normal_predictions)
        false_positive_rate = (false_positives / len(normal_data)) * 100
        
        anomaly_predictions, anomaly_scores = self.predict(anomaly_data, return_scores=True)
        detected_anomalies = np.sum(anomaly_predictions)
        detection_rate = (detected_anomalies / len(anomaly_data)) * 100
        
        print("\nAnomaly Detection Performance:")
        print(f"False Positives: {false_positives} out of {len(normal_data)} normal points ({false_positive_rate:.2f}%)")
        print(f"True Positives: {detected_anomalies} out of {len(anomaly_data)} anomalies ({detection_rate:.2f}%)")
        print("\nReconstruction Error Statistics:")
        print(f"Normal data - Mean: {np.mean(normal_scores):.6f}, Std: {np.std(normal_scores):.6f}")
        print(f"Anomaly data - Mean: {np.mean(anomaly_scores):.6f}, Std: {np.std(anomaly_scores):.6f}")
        
        return {
            'false_positive_rate': false_positive_rate,
            'detection_rate': detection_rate,
            'normal_scores': normal_scores,
            'anomaly_scores': anomaly_scores
        }

def train_model(model, train_data, val_data, epochs=100, batch_size=64):
    """Train model with improved training loop"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for i in range(0, len(train_tensor), batch_size):
            batch = train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            val_loss = criterion(val_outputs, val_tensor)
            
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/(len(train_tensor)//batch_size):.6f}, Val Loss: {val_loss:.6f}')
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def main():
    try:
        # 1. Load and preprocess normal data
        print("Loading and preprocessing data...")
        df = minimal_preprocess('data_MVP.json')
        
        # 2. Prepare features
        features = ['alt', 'gs', 'heading', 'lat', 'lon', 'vertRate', 'altChange_encoded']
        X = df[features].values
        
        # 3. Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 4. Split into train/validation sets
        X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
        
        # 5. Create and train model
        print("\nTraining model...")
        model = FlightPredictionMLP(input_size=len(features))
        model = train_model(model, X_train, X_val)
        
        # 6. Generate synthetic anomalies
        print("\nGenerating synthetic anomalies...")
        anomaly_df = create_synthetic_anomalies(df)
        anomaly_features = scaler.transform(anomaly_df[features].values)
        
        # 7. Create and evaluate anomaly detector
        print("\nEvaluating anomaly detection...")
        detector = AnomalyDetector(model, scaler)
        detector.fit_threshold(X_train)
        
        results = detector.evaluate(X_val, anomaly_features)
        
        # 8. Optional: Detect anomalies in new data
        print("\nExample: Detecting anomalies in sample data...")
        sample_data = X_val[:5]
        predictions = detector.predict(sample_data)
        for i, is_anomaly in enumerate(predictions):
            print(f"Sample {i+1}: {'Anomaly' if is_anomaly else 'Normal'}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()