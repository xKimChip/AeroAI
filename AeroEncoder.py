import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from saliency import SaliencyAnalyzer
from typing import Union, Dict, List
import matplotlib.pyplot as plt
import time #testing saliency



class EnhancedFlightPrediction(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32, 64, 128, 256]):
        """
        Enhanced autoencoder with larger capacity and more sophisticated architecture

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes for the encoder and decoder
        """
        super(EnhancedFlightPrediction, self).__init__()
        self.input_size = input_size

        # Build encoder layers
        encoder_layers = []
        prev_size = input_size
        for h_size in hidden_sizes[:len(hidden_sizes) // 2 + 1]:
            encoder_layers.append(nn.Linear(prev_size, h_size))
            encoder_layers.append(nn.BatchNorm1d(h_size))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.Dropout(0.25))
            prev_size = h_size
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (reverse of encoder)
        decoder_layers = []
        for h_size in hidden_sizes[len(hidden_sizes) // 2 + 1:]:
            decoder_layers.append(nn.Linear(prev_size, h_size))
            decoder_layers.append(nn.BatchNorm1d(h_size))
            decoder_layers.append(nn.LeakyReLU(0.2))
            decoder_layers.append(nn.Dropout(0.25))
            prev_size = h_size
        decoder_layers.append(nn.Linear(prev_size, input_size))
        self.decoder = nn.Sequential(*decoder_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """Get encoded representation"""
        return self.encoder(x)
        
    def add_saliency_analyzer(self, feature_names: List[str]):
        """Initialize saliency analyzer with feature names"""
        self.saliency_analyzer = SaliencyAnalyzer(self, feature_names)
        self._feature_names = feature_names

    def analyze_input(self, x: Union[np.ndarray, torch.Tensor]) -> Dict:
        """Enhanced analysis with numerical saliency values"""
        self.eval()
        
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        with torch.no_grad():
            reconstruction = self.forward(x)
            error = torch.mean((x - reconstruction)**2, dim=1)
            
            results = {
                'input': x.cpu().numpy(),
                'reconstruction': reconstruction.cpu().numpy(),
                'error': error.cpu().numpy()
            }
            
        if hasattr(self, 'saliency_analyzer'):
            try:
                results['explanation'] = self.saliency_analyzer.explain(x)
            except Exception as e:
                print(f"Saliency computation failed: {str(e)}")
                results['explanation'] = None
                
        return results
        
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error"""
        with torch.no_grad():
            return torch.mean((x - self.forward(x))**2, dim=1)

def minimal_preprocess(filepath: str):
    """Minimal preprocessing - only handle format issues"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    initial_count = len(df)
    print(f"\nInitial entries: {initial_count}")

    features = ['alt', 'gs', 'heading', 'lat', 'lon', 'vertRate', 'pitr']  # Added 'pitr' to the conversion list

    # Convert to numeric, keeping all values
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

    # Handle altChange encoding
    df['altChange'] = df['altChange'].fillna(0)

    if 'vertRate' not in df.columns:
        df['vertRate'] = 0

    df['altChange_encoded'] = df['altChange'].map({' ': 0, 'C': 1, 'D': -1}).fillna(0)

    # Ensure time column exists and is numeric
    if 'pitr' in df.columns and 'gs' in df.columns and len(df) > 1:
        # Make sure pitr is numeric before sorting and calculations
        df = df.sort_values(['id', 'pitr']).reset_index(drop=True)

        # Compute time difference in seconds
        df['time_diff'] = df.groupby('id')['pitr'].diff().fillna(1)  # Avoid division by zero

        # Compute rate of change (change per second)
        df['gs_change_rate'] = df.groupby('id')['gs'].diff() / df['time_diff']
        df['heading_change_rate'] = df.groupby('id')['heading'].diff().apply(lambda x: (x + 180) % 360 - 180) / df['time_diff']

        # Replace infinity or extreme values with 0 or NaN
        df['gs_change_rate'] = df['gs_change_rate'].replace([np.inf, -np.inf], 0)  # Replace infinity with 0
        df['heading_change_rate'] = df['heading_change_rate'].replace([np.inf, -np.inf], 0)  # Replace infinity with 0

        # Handle NaN values (if any remain after the replacement) by filling with 0
        df['gs_change_rate'] = df['gs_change_rate'].fillna(0)
        df['heading_change_rate'] = df['heading_change_rate'].fillna(0)

    return df

    def add_saliency_analyzer(self, feature_names: List[str]):
        """Initialize saliency analyzer after model creation"""
        self.saliency_analyzer = SaliencyAnalyzer(self, feature_names)
    
    def analyze_input(self, x: Union[np.ndarray, torch.Tensor]) -> Dict:
        """Enhanced analysis with numerical saliency values"""
        self.eval()
        
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        with torch.no_grad():
            results = {
                'input': x.cpu().numpy(),
                'reconstruction': self.forward(x).cpu().numpy(),
                'error': torch.mean((x - self.forward(x))**2, dim=1).cpu().numpy()
            }
            
        if hasattr(self, 'saliency_analyzer'):
            try:
                results['explanation'] = self.saliency_analyzer.explain(x)
            except Exception as e:
                print(f"Saliency computation failed: {str(e)}")
                results['explanation'] = None
                
        return results

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error"""
        with torch.no_grad():
            x_recon = self.forward(x)
            return torch.mean((x - x_recon)**2, dim=1)


def create_synthetic_anomalies(df, num_anomalies=1000):
    """Generate synthetic anomalies focusing on commercial flight patterns"""
    anomalies = []

    # Define commercial flight characteristics
    COMMERCIAL_MIN_CRUISE_SPEED = 450  # in knots
    COMMERCIAL_MAX_CRUISE_SPEED = 550  # in knots
    COMMERCIAL_MIN_CRUISE_ALT = 30000  # in feet
    COMMERCIAL_MAX_CRUISE_ALT = 40000  # in feet
    COMMERCIAL_NORMAL_VERT_RATE = 2000  # in feet/min
    COMMERCIAL_NORMAL_TURN_RATE = 3  # in degrees/second

    # Define anomaly types with probabilities
    anomaly_types = [
        # Speed anomalies (40% chance)
        {'type': 'high_speed', 'prob': 0.15},
        {'type': 'low_speed_high_alt', 'prob': 0.15},
        {'type': 'sudden_speed_change', 'prob': 0.10},

        # Altitude anomalies (30% chance)
        {'type': 'excessive_altitude', 'prob': 0.10},
        {'type': 'low_altitude_high_speed', 'prob': 0.10},
        {'type': 'rapid_altitude_drop', 'prob': 0.10},

        # Heading anomalies (15% chance)
        {'type': 'impossible_turn', 'prob': 0.05},
        {'type': 'erratic_heading', 'prob': 0.10},

        # Course & pattern anomalies (15% chance)
        # {'type': 'course_deviation', 'prob': 0.05},
        {'type': 'unusual_loitering', 'prob': 0.08},
        {'type': 'altitude_speed_fluctuations', 'prob': 0.07}
    ]

    # Calculate cumulative probabilities
    cum_probs = np.cumsum([a['prob'] for a in anomaly_types])

    for _ in range(num_anomalies):
        # Sample a base flight
        base_flight = df.sample(n=1).iloc[0].to_dict()
        anomaly = base_flight.copy()

        # Randomly select an anomaly type based on probabilities
        rand_val = np.random.random()
        anomaly_idx = np.searchsorted(cum_probs, rand_val)
        anomaly_type = anomaly_types[anomaly_idx]['type']

        # Apply the selected anomaly
        if anomaly_type == 'high_speed':
            anomaly['anomaly_type'] = anomaly_type
            # Commercial aircraft flying faster than 600 knots
            anomaly['gs'] = np.random.uniform(600, 1000)
            # anomaly['gs_change'] = np.random.uniform(50, 100)  # Accelerating
            anomalies.append(anomaly)

        elif anomaly_type == 'low_speed_high_alt':
            anomaly['anomaly_type'] = anomaly_type
            # Aircraft flying below 200 knots at high altitude (stall risk)
            anomaly['gs'] = np.random.uniform(100, 200)
            anomaly['alt'] = np.random.uniform(35000, 40000)
            # anomaly['gs_change'] = np.random.uniform(-50, -20)  # Decelerating
            anomalies.append(anomaly)

        elif anomaly_type == 'sudden_speed_change':
            anomaly['anomaly_type'] = anomaly_type
            # Sudden ±200 knots change (not physically possible)
            direction = np.random.choice([-1, 1])
            speed_change = np.random.uniform(200, 250) * direction

            anomaly['gs'] = max(0, base_flight['gs'] + speed_change)  # Ensure non-negative speed
            time_diff = np.random.uniform(1, 3)
            # Ensure 'time_diff' exists and is valid
            anomaly['gs_change_rate'] = speed_change / time_diff
            anomalies.append(anomaly)


        elif anomaly_type == 'excessive_altitude':
            anomaly['anomaly_type'] = anomaly_type
            # Commercial aircraft flying above 45,000 ft
            anomaly['alt'] = np.random.uniform(45000, 60000)
            # anomaly['vertRate'] = np.random.uniform(1000, 2000)  # Still climbing
            anomalies.append(anomaly)

        elif anomaly_type == 'low_altitude_high_speed':
            anomaly['anomaly_type'] = anomaly_type
            # High speed at low altitude
            anomaly['alt'] = np.random.uniform(5000, 10000)
            anomaly['gs'] = np.random.uniform(450, 550)
            # anomaly['vertRate'] = np.random.uniform(-1000, 1000)  # Level or slight descent/climb
            anomalies.append(anomaly)

        elif anomaly_type == 'rapid_altitude_drop':
            anomaly['anomaly_type'] = anomaly_type
            # Rapid altitude drop (possible emergency)
            anomaly['vertRate'] = np.random.uniform(-8000, -5000)
            # anomaly['gs_change'] = np.random.uniform(-50, 50)  # Possible speed changes during emergency
            anomalies.append(anomaly)

        elif anomaly_type == 'impossible_turn':
            anomaly['anomaly_type'] = anomaly_type
            # Physically impossible turns for aircraft
            direction = np.random.choice([-1, 1])  # Left (-1) or right (+1)
            heading_change = direction * np.random.uniform(90, 120)  # Extreme turn rate

            anomaly['heading'] = (base_flight['heading'] + heading_change) % 360
            # Use a random time difference (1 to 5 seconds) to ensure a reasonable heading change rate
            time_diff = np.random.uniform(1, 5)  # Time in seconds
            anomaly['heading_change_rate'] = heading_change / time_diff  # Degrees per second
            anomalies.append(anomaly)

        elif anomaly_type == 'erratic_heading':
            anomaly['anomaly_type'] = anomaly_type
            # Create zigzag pattern in heading
            direction = np.random.choice([-1, 1])  # Left (-1) or right (+1)
            heading_change = direction * np.random.uniform(40, 80)  # Random change in range

            anomaly['heading'] = (base_flight['heading'] + heading_change) % 360
            time_diff = np.random.uniform(1, 2)
            anomaly['heading_change_rate'] = heading_change / time_diff
            anomalies.append(anomaly)

        # elif anomaly_type == 'course_deviation':
        #     anomaly['anomaly_type'] = anomaly_type
        #     # Significant deviation from expected route
        #     deviation_magnitude = np.random.uniform(0.3, 0.7)  # roughly 20-40 NM
        #     anomaly['lat'] = float(base_flight['lat']) + deviation_magnitude * np.random.choice([-1, 1])
        #     anomaly['lon'] = float(base_flight['lon']) + deviation_magnitude * np.random.choice([-1, 1])

        #     time_diff = np.random.uniform(1, 2)  # Time in seconds
        #     heading_change = np.random.uniform(20, 40)  # Course correction
        #     anomaly['heading_change_rate'] = heading_change / time_diff  # Degrees per second
        #     anomalies.append(anomaly)

        elif anomaly_type == 'unusual_loitering':
            anomaly['anomaly_type'] = anomaly_type
            # Aircraft circling or loitering unexpectedly
            anomaly['gs'] = np.random.uniform(50, 100)
            anomaly['vertRate'] = np.random.uniform(-100, 100)

            time_diff = np.random.uniform(1, 2)  # Time in seconds
            heading_change = np.random.uniform(5, 15)  # Slight turning
            anomaly['heading_change_rate'] = heading_change / time_diff  # Degrees per second
            anomalies.append(anomaly)

        elif anomaly_type == 'altitude_speed_fluctuations':
            time_diff = np.random.uniform(1, 5)  # Time in seconds
            if np.random.random() > 0.5:
                anomaly['anomaly_type'] = anomaly_type
                # Climbing rapidly while speeding up
                anomaly['vertRate'] = np.random.uniform(3000, 4000)
                speed_change = np.random.uniform(50, 100)
                anomaly['gs'] = float(base_flight['gs']) + speed_change
                anomaly['gs_change_rate'] = speed_change / time_diff  # Knots per second
            else:
                # Descending rapidly while slowing down
                anomaly['vertRate'] = np.random.uniform(-4000, -3000)
                speed_change = np.random.uniform(50, 100)
                anomaly['gs'] = max(0, float(base_flight['gs']) - speed_change)  # Ensure non-negative speed
                anomaly['gs_change_rate'] = -speed_change / time_diff  # Knots per second (negative for slowing down)

                # Add label for what type of anomaly this is (for analysis)
                anomaly['anomaly_type'] = anomaly_type
            anomalies.append(anomaly)

    anomaly_df = pd.DataFrame(anomalies)

    # Print statistics about the generated anomalies
    anomaly_counts = anomaly_df['anomaly_type'].value_counts()
    # print("\nSynthetic Commercial Flight Anomalies Created:")
    # for anomaly_type, count in anomaly_counts.items():
    #     print(f"  - {anomaly_type}: {count} instances ({count/num_anomalies*100:.1f}%)")

    return anomaly_df


class AnomalyDetector:
    def __init__(self, model, scaler, threshold_multiplier=2.5):
        self.model = model
        self.scaler = scaler
        self.threshold = None
        self.threshold_multiplier = threshold_multiplier

    def fit_threshold(self, normal_data):
        """Determine threshold using normal data with enhanced approach"""
        self.model.eval()
        with torch.no_grad():
            normal_tensor = torch.FloatTensor(normal_data)
            predictions = self.model(normal_tensor)
            reconstruction_errors = torch.mean((normal_tensor - predictions) ** 2, dim=1)

            # Use more sophisticated threshold calculation
            # Based on percentile rather than just mean + std
            sorted_errors = torch.sort(reconstruction_errors)[0]
            percentile_95 = sorted_errors[int(0.95 * len(sorted_errors))]
            percentile_99 = sorted_errors[int(0.99 * len(sorted_errors))]

            # Use a weighted combination of statistics
            self.threshold = 0.5 * percentile_95 + 0.5 * percentile_99

            # print(f"\nSet anomaly threshold at: {self.threshold:.6f}")
            # print(f"95th percentile: {percentile_95:.6f}")
            # print(f"99th percentile: {percentile_99:.6f}")

    def predict(self, data, return_scores=False, altitude_threshold=45000.0, speed_threshold=600.0):
        """Predict anomalies in new data"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(data, pd.DataFrame):
                data_np = data.values
            else:
                data_np = np.array(data)
            data_tensor = torch.FloatTensor(data)
            predictions = self.model(data_tensor)
            reconstruction_errors = torch.mean((data_tensor - predictions) ** 2, dim=1)

            anomalies = reconstruction_errors > self.threshold

            altitude_col = 0  # Altitude column index

            altitude_constraint = torch.tensor(
                data_np[:, altitude_col] > (altitude_threshold - self.scaler.mean_[altitude_col]) / self.scaler.scale_[
                    altitude_col],
                dtype=torch.bool
            )
            anomalies |= altitude_constraint

            speed_col = 2  # Ground speed column index

            # Apply ground speed constraint if column is specified
            speed_constraint = torch.tensor(
                data_np[:, speed_col] > (speed_threshold - self.scaler.mean_[speed_col]) / self.scaler.scale_[
                    speed_col],
                dtype=torch.bool
            )
            anomalies |= speed_constraint

            anomalies_np = anomalies.numpy()
            #print(predictions)

            # Create a DataFrame with the original data
            if isinstance(data, pd.DataFrame):
                data_df = data.copy()
                data_df['anomaly'] = anomalies_np
            else:
                data_df = pd.DataFrame(data_np)
                data_df['anomaly'] = anomalies_np

            #print(data_df['anomaly'])

            if return_scores:
                return anomalies_np, reconstruction_errors.numpy()
            else:
                # Create a DataFrame with the original data
                if isinstance(data, pd.DataFrame):
                    data_df = data.copy()
                    data_df['anomaly'] = anomalies_np
                else:
                    data_df = pd.DataFrame(data_np)
                    data_df['anomaly'] = anomalies_np

                return data_df


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

        # Analyze performance by anomaly type
        if 'anomaly_type' in anomaly_data:
            # print("\nDetection Rate by Anomaly Type:")
            anomaly_df = pd.DataFrame({
                'anomaly_type': anomaly_data['anomaly_type'],
                'is_detected': anomaly_predictions,
                'score': anomaly_scores
            })

            for anomaly_type, group in anomaly_df.groupby('anomaly_type'):
                detection_rate = (group['is_detected'].sum() / len(group)) * 100
                avg_score = group['score'].mean()
                # print(f"  - {anomaly_type}: {detection_rate:.2f}% detected (avg score: {avg_score:.6f})")

        return {
            'false_positive_rate': false_positive_rate,
            'detection_rate': detection_rate,
            'normal_scores': normal_scores,
            'anomaly_scores': anomaly_scores
        }
        def explain_anomaly(self, 
                      x: Union[np.ndarray, torch.Tensor],
                      method: str = 'saliency') -> Dict:
            """
            Explain why a sample was flagged as anomalous
            Args:
                x: Input data (single sample only)
                method: Saliency method to use
            Returns:
                Explanation dictionary with:
                - top_features: Most contributing features
                - saliency_map: Raw saliency values
                - reconstruction_error: MSE error
            """
            if not hasattr(self.model, 'saliency_analyzer'):
                raise RuntimeError("Initialize saliency analyzer first with add_saliency_analyzer()")
            
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Ensure batch dimension
                
            self.model.eval()
            with torch.no_grad():
                error = self.model.get_reconstruction_error(x)
                
            explanation = self.model.saliency_analyzer.explain(x, method=method)
            explanation['reconstruction_error'] = error.item()
            
            return explanation

def train_model(model, train_data, val_data, epochs=150, batch_size=128):
    """Train model with improved training loop and learning rate scheduling"""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Multi-step learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 60, 90, 120],
        gamma=0.5
    )

    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)

    # For early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience = 25  # Increased patience for larger model
    patience_counter = 0

    # Training history
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        batch_count = 0

        # Create random permutation for batch sampling
        indices = torch.randperm(len(train_tensor))

        for i in range(0, len(train_tensor), batch_size):
            # Get batch using permutation
            batch_indices = indices[i:i+batch_size]
            batch = train_tensor[batch_indices]

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            val_loss = criterion(val_outputs, val_tensor)

        # Learning rate scheduling
        scheduler.step()

        # Save losses for plotting
        avg_train_loss = total_loss / batch_count
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss.item())

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            # print(f"Early stopping at epoch {epoch+1}")
            break

        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses

def save_model(model, scaler, detector, save_dir='saved_models'):
    """Save the trained model, scaler, and detector configuration"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, 'flight_anomaly_model.pth')
    scaler_path = os.path.join(save_dir, 'scaler.joblib')
    detector_path = os.path.join(save_dir, 'detector.joblib')

    # Save the PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.input_size,
        'model_type': 'EnhancedFlightPrediction'
    }, model_path)

    # Save the scaler
    joblib.dump(scaler, scaler_path)

    # Save detector configuration
    detector_config = {
        'threshold': detector.threshold.item() if detector.threshold is not None else None,
        'threshold_multiplier': detector.threshold_multiplier
    }
    joblib.dump(detector_config, detector_path)

    # print(f"\nModel saved successfully")

def load_model(model_path='saved_models/flight_anomaly_model.pth',
              scaler_path='saved_models/scaler.joblib',
              detector_path='saved_models/detector.joblib'):
    """Load the saved model, scaler, and detector configuration"""
    # Load model
    checkpoint = torch.load(model_path)
    model_type = checkpoint.get('model_type', 'FlightPrediction')

    if model_type == 'EnhancedFlightPrediction':
        model = EnhancedFlightPrediction(input_size=checkpoint['input_size'])
    else:
        # Fallback for compatibility with older models
        from collections import OrderedDict
        model = FlightPrediction(input_size=checkpoint['input_size'])

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load detector configuration
    detector_config = joblib.load(detector_path)
    detector = AnomalyDetector(model, scaler, detector_config['threshold_multiplier'])
    detector.threshold = torch.tensor(detector_config['threshold'])

    # print(f"\nModel loaded successfully: {model_type}")
    return model, scaler, detector


def test_additional_data(model, scaler, detector, features, num_runs=1000):
    """
    Load and test data from multiple files, insert synthetic anomalies,
    and calculate average true/false positive rates over multiple runs for each file
    """
    print(f"\n===== Testing Additional Data Files with Synthetic Anomalies ({num_runs} runs) =====")

    # Define paths to the test files
    test_file_paths = [
        'data/DFW_hose_data.json',
        'data/Israel_data.json',
        'data/FRA_hose_data.json'
    ]

    # Create a dictionary to store aggregated metrics for each file
    aggregated_metrics = {}

    for i, file_path in enumerate(test_file_paths):
        try:
            file_name = os.path.basename(file_path)
            print(f"\nProcessing test file {i + 1}: {file_name}")

            # Load and preprocess the test file
            test_df = minimal_preprocess(file_path)
            print(f"Loaded {len(test_df)} entries from test file {i + 1}")

            # Ensure all required features exist
            missing_features = [f for f in features if f not in test_df.columns]
            if missing_features:
                print(f"Warning: Missing features in test file {i + 1}: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    if feature == 'altChange_encoded':
                        test_df[feature] = 0
                    elif feature == 'vertRate':
                        test_df[feature] = 0
                    elif feature in ['gs_change_rate', 'heading_change_rate']:
                        test_df[feature] = 0

            # Initialize metrics storage for this file
            file_metrics = {
                'true_positive_rate': [],
                'false_positive_rate': [],
                'precision': [],
                'f1_score': [],
                'accuracy': [],
                'anomaly_type_detection': {}
            }

            # Run multiple times to get average performance
            for run in range(num_runs):
                print(f"\nRun {run + 1}/{num_runs} for file {i + 1}")

                # Create a copy of the original data to use as "normal" data for evaluation
                normal_df = test_df.copy()

                # Generate synthetic anomalies based on this test file
                num_anomalies = int(len(test_df) * 0.1)  # Create anomalies equal to 10% of test data
                print(f"Generating {num_anomalies} synthetic anomalies...")
                anomaly_df = create_synthetic_anomalies(test_df, num_anomalies=num_anomalies)

                # Add a label column to track which rows are actual anomalies
                normal_df['actual_anomaly'] = 0
                anomaly_df['actual_anomaly'] = 1

                # Combine normal and anomaly data
                combined_df = pd.concat([normal_df, anomaly_df], ignore_index=True)

                # Extract features for anomaly detection
                combined_features = combined_df[features].values

                # Scale the features
                combined_features_scaled = scaler.transform(combined_features)

                # Run anomaly detection
                with torch.no_grad():
                    data_tensor = torch.FloatTensor(combined_features_scaled)
                    predictions = model(data_tensor)
                    reconstruction_errors = torch.mean((data_tensor - predictions) ** 2, dim=1)
                    predicted_anomalies = reconstruction_errors > detector.threshold
                    predicted_anomalies_np = predicted_anomalies.numpy()

                # Add predictions to the combined dataframe
                combined_df['predicted_anomaly'] = predicted_anomalies_np

                # Calculate performance metrics
                tp = sum((combined_df['actual_anomaly'] == 1) & (combined_df['predicted_anomaly'] == 1))
                fp = sum((combined_df['actual_anomaly'] == 0) & (combined_df['predicted_anomaly'] == 1))
                tn = sum((combined_df['actual_anomaly'] == 0) & (combined_df['predicted_anomaly'] == 0))
                fn = sum((combined_df['actual_anomaly'] == 1) & (combined_df['predicted_anomaly'] == 0))

                # Calculate rates
                true_positive_rate = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                false_positive_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
                precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
                f1_score = 2 * (precision * true_positive_rate) / (precision + true_positive_rate) if (
                                                                                                                  precision + true_positive_rate) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn) * 100

                # Store metrics for this run
                file_metrics['true_positive_rate'].append(true_positive_rate)
                file_metrics['false_positive_rate'].append(false_positive_rate)
                file_metrics['precision'].append(precision)
                file_metrics['f1_score'].append(f1_score)
                file_metrics['accuracy'].append(accuracy)

                # Analyze performance by anomaly type
                if 'anomaly_type' in combined_df.columns:
                    for anomaly_type, group in combined_df[combined_df['actual_anomaly'] == 1].groupby('anomaly_type'):
                        group_tp = sum(group['predicted_anomaly'] == 1)
                        detection_rate = (group_tp / len(group)) * 100

                        if anomaly_type not in file_metrics['anomaly_type_detection']:
                            file_metrics['anomaly_type_detection'][anomaly_type] = []

                        file_metrics['anomaly_type_detection'][anomaly_type].append(detection_rate)

                # Save the last run's results to CSV for reference
                if run == num_runs - 1:
                    result_file = f'/Users/dush/PycharmProjects/Raytheon/AeroAI/results/{file_name}_results_run{run + 1}.csv'
                    os.makedirs(os.path.dirname(result_file), exist_ok=True)
                    combined_df.to_csv(result_file, index=False)
                    print(f"  - Final run results saved to: {result_file}")

            # Calculate and display average metrics for this file
            avg_tpr = np.mean(file_metrics['true_positive_rate'])
            avg_fpr = np.mean(file_metrics['false_positive_rate'])
            avg_precision = np.mean(file_metrics['precision'])
            avg_f1 = np.mean(file_metrics['f1_score'])
            avg_accuracy = np.mean(file_metrics['accuracy'])

            std_tpr = np.std(file_metrics['true_positive_rate'])
            std_fpr = np.std(file_metrics['false_positive_rate'])
            std_precision = np.std(file_metrics['precision'])
            std_f1 = np.std(file_metrics['f1_score'])
            std_accuracy = np.std(file_metrics['accuracy'])

            print(f"\nAverage Performance Metrics for {file_name} over {num_runs} runs:")
            print(f"  - True Positive Rate (Recall): {avg_tpr:.2f}% ± {std_tpr:.2f}%")
            print(f"  - False Positive Rate: {avg_fpr:.2f}% ± {std_fpr:.2f}%")
            print(f"  - Precision: {avg_precision:.2f}% ± {std_precision:.2f}%")
            print(f"  - F1 Score: {avg_f1:.2f}% ± {std_f1:.2f}%")
            print(f"  - Accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")

            # Display average performance by anomaly type for this file
            if file_metrics['anomaly_type_detection']:
                print("\nAverage Performance by Anomaly Type:")
                for anomaly_type, rates in file_metrics['anomaly_type_detection'].items():
                    avg_rate = np.mean(rates)
                    std_rate = np.std(rates)
                    print(f"  - {anomaly_type}: {avg_rate:.2f}% ± {std_rate:.2f}% detected")

            # Store the aggregated metrics for this file
            aggregated_metrics[file_name] = {
                'true_positive_rate': (avg_tpr, std_tpr),
                'false_positive_rate': (avg_fpr, std_fpr)
            }

            # Save the metrics for this file to a separate JSON file
            metrics_file = f'/Users/dush/PycharmProjects/Raytheon/AeroAI/results/{file_name}_test_metrics.json'

            # Convert numpy values to Python native types for JSON serialization
            json_metrics = {
                'true_positive_rate': (float(avg_tpr), float(std_tpr)),
                'false_positive_rate': (float(avg_fpr), float(std_fpr))
            }

            with open(metrics_file, 'w') as f:
                json.dump(json_metrics, f, indent=2)

            print(f"  - Metrics for {file_name} saved to: {metrics_file}")

        except Exception as e:
            print(f"Error processing test file {i + 1}: {str(e)}")

    return aggregated_metrics
def analyze_anomaly_scores(anomaly_df, scores, detector_threshold):
    """Analyze detection performance for different types of anomalies"""
    anomaly_df['score'] = scores
    anomaly_df['is_detected'] = scores > detector_threshold

    # print("\nAnomaly Detection Analysis by Type:")
    for anomaly_type, group in anomaly_df.groupby('anomaly_type'):
        detection_rate = (group['is_detected'].sum() / len(group)) * 100
        avg_score = group['score'].mean()
        max_score = group['score'].max()
        min_score = group['score'].min()

        print(f"\n{anomaly_type.upper()}:")
        print(f"  - Detection rate: {detection_rate:.2f}%")
        print(f"  - Average score: {avg_score:.6f}")
        print(f"  - Score range: {min_score:.6f} to {max_score:.6f}")

        # Find hardest to detect examples
        if not group['is_detected'].all() and len(group) > 1:
            missed = group[~group['is_detected']].sort_values('score', ascending=True)
            # if len(missed) > 0:
            #     print(f"  - Hardest to detect example (score: {missed.iloc[0]['score']:.6f}):")
            #     for feature in ['alt', 'gs', 'heading', 'vertRate','gs_change_rate','heading_change_rate']:
            #         if feature in missed.columns:
            #             print(f"    {feature}: {missed.iloc[0][feature]}")

# def flight_prediction(data, model, scaler, detector):
#     """Predict anomalies in flight data using the trained model"""
#     #features = ['alt', 'gs', 'heading', 'lat', 'lon', 'vertRate', 'altChange_encoded', 'gs_change_rate', 'heading_change_rate']
#     features = ['alt', 'gs', 'heading', 'vertRate', 'altChange_encoded', 'gs_change_rate', 'heading_change_rate']
    
#     X = data[features].values
#     scaled = scaler.transform(X)
#     anomalies = detector.predict(scaled)
#     return anomalies

def flight_prediction(data, model, scaler, detector, explain=False):
    features = ['alt', 'gs', 'heading', 'vertRate', 'altChange_encoded', 'gs_change_rate', 'heading_change_rate']
    scaled = scaler.transform(data[features].values)
    
    anomalies_df = detector.predict(scaled)

    if explain:
        analysis = model.analyze_input(torch.FloatTensor(scaled))
        anomalies_df['saliency'] = analysis.get('explanation', None)

    return anomalies_df


def evaluate_attribution_accuracy(anomaly_explanations, anomaly_df, features):
    """
    Evaluate if the saliency maps correctly identify the features that were modified
    to create each anomaly type.
    
    Args:
        anomaly_explanations: List of explanations from saliency analysis
        anomaly_df: DataFrame containing the anomalies with their types
        features: List of feature names
        
    Returns:
        Dictionary of accuracy metrics by anomaly type
    """
    # Define which features were modified for each anomaly type
    anomaly_feature_map = {
        'high_speed': ['gs'],  # High speed anomalies modify ground speed
        'low_speed_high_alt': ['gs', 'alt'],  # Low speed at high altitude modifies both
        'sudden_speed_change': ['gs', 'gs_change_rate'],  # Sudden speed changes modify GS and its rate
        'excessive_altitude': ['alt'],  # Excessive altitude modifies altitude
        'low_altitude_high_speed': ['alt', 'gs'],  # Low altitude with high speed modifies both
        'rapid_altitude_drop': ['vertRate'],  # Rapid altitude drop modifies vertical rate
        'impossible_turn': ['heading', 'heading_change_rate'],  # Impossible turns modify heading and its rate
        'erratic_heading': ['heading', 'heading_change_rate'],  # Erratic heading modifies heading and its rate
        'unusual_loitering': ['gs', 'vertRate', 'heading_change_rate'],  # Loitering modifies GS, vertical rate, and heading
        'altitude_speed_fluctuations': ['vertRate', 'gs', 'gs_change_rate']  # Fluctuations modify vertical rate and GS
    }
    
    # Initialize counters for each anomaly type
    attribution_metrics = {}
    for anomaly_type in anomaly_feature_map.keys():
        attribution_metrics[anomaly_type] = {
            'total': 0,
            'detected': 0,
            'correctly_attributed': 0,
            'attribution_accuracy': 0.0
        }
    
    # Process each anomaly explanation
    for explanation in anomaly_explanations:
        anomaly_type = explanation['anomaly_type']
        
        # Skip unknown anomaly types
        if anomaly_type not in anomaly_feature_map:
            continue
            
        # Count this anomaly
        attribution_metrics[anomaly_type]['total'] += 1
        
        # Count as detected
        attribution_metrics[anomaly_type]['detected'] += 1
        
        # Check if saliency correctly identifies modified features
        if 'saliency' in explanation and 'top_features' in explanation['saliency']:
            # Extract top feature names (without importance values)
            top_feature_names = [feat for feat, _ in explanation['saliency']['top_features'][:2]]
            
            # Get the expected modified features for this anomaly type
            expected_features = anomaly_feature_map[anomaly_type]
            
            # Check if ANY of the expected features are in the top 2
            attribution_correct = any(feat in top_feature_names for feat in expected_features)
            
            if attribution_correct:
                attribution_metrics[anomaly_type]['correctly_attributed'] += 1
    
    # Calculate attribution accuracy for each anomaly type
    for anomaly_type in attribution_metrics:
        detected = attribution_metrics[anomaly_type]['detected']
        if detected > 0:
            correctly_attributed = attribution_metrics[anomaly_type]['correctly_attributed']
            attribution_metrics[anomaly_type]['attribution_accuracy'] = (correctly_attributed / detected) * 100
    
    return attribution_metrics

def print_attribution_report(attribution_metrics):
    """
    Print a detailed report of detection rates and attribution accuracy
    
    Args:
        attribution_metrics: Dictionary of accuracy metrics by anomaly type
    """
    print("\n===== ANOMALY DETECTION AND ATTRIBUTION ACCURACY REPORT =====")
    
    for anomaly_type, metrics in attribution_metrics.items():
        total = metrics['total']
        detected = metrics['detected']
        correctly_attributed = metrics['correctly_attributed']
        
        # Calculate rates
        detection_rate = (detected / total) * 100 if total > 0 else 0
        attribution_accuracy = metrics['attribution_accuracy']
        
        # Print the report for this anomaly type
        print(f"\n{anomaly_type}:")
        print(f"  Detection rate: {detection_rate:.2f}% ({detected}/{total})")
        print(f"  Attribution accuracy: {attribution_accuracy:.2f}% ({correctly_attributed}/{detected})")

def analyze_individual_anomalies(anomaly_df, model, scaler, detector, features):
    """
    Run saliency analysis on each individual anomaly and evaluate attribution accuracy
    
    Args:
        anomaly_df: DataFrame containing synthetic anomalies
        model: Trained autoencoder model
        scaler: Fitted scaler for data normalization
        detector: Anomaly detector with set threshold
        features: List of feature names
        
    Returns:
        Tuple of (anomaly_explanations, attribution_metrics)
    """
    # Ensure model has saliency analyzer
    if not hasattr(model, 'saliency_analyzer'):
        model.add_saliency_analyzer(features)
    
    # Scale anomaly features
    anomaly_features = anomaly_df[features].values
    anomaly_features_scaled = scaler.transform(anomaly_features)
    
    # Detect anomalies
    results = detector.predict(anomaly_features_scaled, return_scores=True)
    is_anomaly = results[0]  # Boolean array of detected anomalies
    scores = results[1]      # Reconstruction error scores
    
    print(f"Found {sum(is_anomaly)} anomalies out of {len(anomaly_df)} samples")
    
    # Process each detected anomaly
    anomaly_explanations = []
    for idx, (is_detected, score) in enumerate(zip(is_anomaly, scores)):
        if is_detected:
            # Get the anomaly features and scale them
            anomaly = anomaly_df.iloc[idx]
            anomaly_scaled = scaler.transform(anomaly[features].values.reshape(1, -1))
            
            # Run saliency analysis
            analysis = model.analyze_input(torch.FloatTensor(anomaly_scaled))
            
            # Store explanation along with anomaly info
            explanation = {
                'index': idx,
                'anomaly_type': anomaly.get('anomaly_type', 'Unknown'),
                'reconstruction_error': score,
                'features': {feat: anomaly[feat] for feat in features},
                'saliency': analysis.get('explanation', {})
            }
            anomaly_explanations.append(explanation)
    
    # Evaluate attribution accuracy
    attribution_metrics = evaluate_attribution_accuracy(anomaly_explanations, anomaly_df, features)
    
    # Print the attribution accuracy report
    print_attribution_report(attribution_metrics)
    
    return anomaly_explanations, attribution_metrics

def main():
    try:
        # 1. Load and preprocess normal data
        print("Loading and preprocessing data...")
        df = minimal_preprocess('data/FA-KSA_recording.json')

        # 2. Prepare features
        features = ['alt', 'gs', 'heading', 'vertRate', 'altChange_encoded']
        if 'gs_change_rate' in df.columns:
            features.append('gs_change_rate')
        if 'heading_change_rate' in df.columns:
            features.append('heading_change_rate')

        X = df[features].values

        # 3. Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("\nValues Scaled")

        # 4. Split into train/validation sets
        X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

        # 5. Create and train the enhanced model
        print("\nTraining enhanced model with larger architecture...")
        model = EnhancedFlightPrediction(input_size=len(features))
        model, train_losses, val_losses = train_model(model, X_train, X_val, epochs=150, batch_size=128)

        # 6. Generate synthetic anomalies
        print("\nGenerating synthetic commercial flight anomalies...")
        anomaly_df = create_synthetic_anomalies(df, num_anomalies=1000)
        
        # 7. Create and evaluate detector
        detector = AnomalyDetector(model, scaler, threshold_multiplier=2.5)
        detector.fit_threshold(X_train)

        # 8. Initialize saliency analyzer
        model.add_saliency_analyzer(features)
        
        # 9. Analyze all anomalies individually with attribution accuracy
        print("\nAnalyzing individual anomalies with attribution accuracy...")
        anomaly_explanations, attribution_metrics = analyze_individual_anomalies(
            anomaly_df, model, scaler, detector, features
        )
        
        # 10. Save results to file (optional)
        results = {
            'explanations': anomaly_explanations,
            'attribution_metrics': attribution_metrics
        }
        
        # with open('anomaly_analysis_results.json', 'w') as f:
        #     json.dump(results, f, indent=2)
            
        print("\nEnhanced model training and evaluation complete!")
        print("Attribution accuracy results saved to 'anomaly_analysis_results.json'")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()