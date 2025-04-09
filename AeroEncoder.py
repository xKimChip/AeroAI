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

class EnhancedFlightPrediction(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32, 64, 128, 256]):
        """
        Enhanced autoencoder with larger capacity and more sophisticated architecture

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes for the encoder and decoder
        """
        super(EnhancedFlightPrediction, self).__init__()

        # Ensure symmetric architecture for proper autoencoder
        self.input_size = input_size

        # Build encoder layers
        encoder_layers = []
        prev_size = input_size
        for h_size in hidden_sizes[:len(hidden_sizes)//2 + 1]:
            encoder_layers.append(nn.Linear(prev_size, h_size))
            encoder_layers.append(nn.BatchNorm1d(h_size))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.Dropout(0.25))
            prev_size = h_size

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (reverse of encoder)
        decoder_layers = []
        for h_size in hidden_sizes[len(hidden_sizes)//2 + 1:]:
            decoder_layers.append(nn.Linear(prev_size, h_size))
            decoder_layers.append(nn.BatchNorm1d(h_size))
            decoder_layers.append(nn.LeakyReLU(0.2))
            decoder_layers.append(nn.Dropout(0.25))
            prev_size = h_size

        # Final layer to reconstruct input
        decoder_layers.append(nn.Linear(prev_size, input_size))

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights for better gradient flow
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

def minimal_preprocess(filepath: str):
    """Minimal preprocessing - only handle format issues"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame([data])
    
    initial_count = len(df)
    # print(f"\nInitial entries: {initial_count}")

    features = ['alt', 'gs', 'heading', 'vertRate']

    # Convert to numeric, keeping all values
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                # print(f"Found {nan_count} missing/invalid values in {col}")
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                # print(f"Filled missing values with median: {median_val}")

    # Handle altChange encoding

    
    df['altChange'] = df['altChange'].fillna(0)
    if 'vertRate' not in df.columns:
        df['vertRate'] = 0
    
    df['altChange_encoded'] = df['altChange'].map({' ': 0, 'C': 1, 'D': -1}).fillna(0)

    

    # Ensure time column exists
    if 'pitr' in df.columns and 'gs' in df.columns and len(df) > 1:
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

    # print(f"\nFeature engineering complete")
    # print(df)

    # print(f"\nFinal entries: {len(df)}")
    return df

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
        #{'type': 'course_deviation', 'prob': 0.05},
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
            #anomaly['gs_change'] = np.random.uniform(50, 100)  # Accelerating
            anomalies.append(anomaly)

        elif anomaly_type == 'low_speed_high_alt':
            anomaly['anomaly_type'] = anomaly_type
            # Aircraft flying below 200 knots at high altitude (stall risk)
            anomaly['gs'] = np.random.uniform(100, 200)
            anomaly['alt'] = np.random.uniform(35000, 40000)
            #anomaly['gs_change'] = np.random.uniform(-50, -20)  # Decelerating
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
            #anomaly['vertRate'] = np.random.uniform(1000, 2000)  # Still climbing
            anomalies.append(anomaly)

        elif anomaly_type == 'low_altitude_high_speed':
            anomaly['anomaly_type'] = anomaly_type
            # High speed at low altitude
            anomaly['alt'] = np.random.uniform(5000, 10000)
            anomaly['gs'] = np.random.uniform(450, 550)
            #anomaly['vertRate'] = np.random.uniform(-1000, 1000)  # Level or slight descent/climb
            anomalies.append(anomaly)

        elif anomaly_type == 'rapid_altitude_drop':
            anomaly['anomaly_type'] = anomaly_type
            # Rapid altitude drop (possible emergency)
            anomaly['vertRate'] = np.random.uniform(-8000, -5000)
            #anomaly['gs_change'] = np.random.uniform(-50, 50)  # Possible speed changes during emergency
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

    def predict(self, data, return_scores=False,altitude_threshold=45000.0,speed_threshold=600.0):
        """Predict anomalies in new data"""
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data)
            predictions = self.model(data_tensor)
            reconstruction_errors = torch.mean((data_tensor - predictions) ** 2, dim=1)

            anomalies = reconstruction_errors > self.threshold

            altitude_col = 0  # Altitude column index

            altitude_constraint = torch.from_numpy(
                data[:, altitude_col] > (altitude_threshold - self.scaler.mean_[altitude_col]) / self.scaler.scale_[altitude_col]
            )
            anomalies |= altitude_constraint
            
            # print("TESTING SCALER ALTITUDE")
            # print(self.scaler.mean_[altitude_col])
            # print(self.scaler.scale_[altitude_col])

            speed_col = 2  # Ground speed column index

            # Apply ground speed constraint if column is specified
            speed_constraint = torch.from_numpy(
                data[:, speed_col] > (speed_threshold - self.scaler.mean_[speed_col]) / self.scaler.scale_[speed_col]
            )
            anomalies |= speed_constraint

            # print("TESTING SCALER SPEED")
            # print(self.scaler.mean_[speed_col])
            # print(self.scaler.scale_[speed_col])
            
            anomalies_np = anomalies.numpy()
            print(predictions)
            #anomalies_df = pd.DataFrame(anomalies_np )
            
            features = ['alt', 'gs', 'heading', 'lat', 'lon', 'vertRate', 'altChange_encoded']
            data_df = pd.DataFrame(data)
            
            data_df['anomaly'] = anomalies_np
            #result = np.hstack((data, anomalies_np))
            print(data_df['anomaly'])
            
            #data['anomaly'] = anomalies_np

            if return_scores:
                data['reconstruction_error'] = reconstruction_errors.numpy()
                return data
            return data_df

    def evaluate(self, normal_data, anomaly_data):
        """Evaluate detector performance"""
        normal_predictions, normal_scores = self.predict(normal_data, return_scores=True)
        false_positives = np.sum(normal_predictions)
        false_positive_rate = (false_positives / len(normal_data)) * 100

        anomaly_predictions, anomaly_scores = self.predict(anomaly_data, return_scores=True)
        detected_anomalies = np.sum(anomaly_predictions)
        detection_rate = (detected_anomalies / len(anomaly_data)) * 100

        # print("\nAnomaly Detection Performance:")
        # print(f"False Positives: {false_positives} out of {len(normal_data)} normal points ({false_positive_rate:.2f}%)")
        # print(f"True Positives: {detected_anomalies} out of {len(anomaly_data)} anomalies ({detection_rate:.2f}%)")
        # print("\nReconstruction Error Statistics:")
        # print(f"Normal data - Mean: {np.mean(normal_scores):.6f}, Std: {np.std(normal_scores):.6f}")
        # print(f"Anomaly data - Mean: {np.mean(anomaly_scores):.6f}, Std: {np.std(anomaly_scores):.6f}")

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

        # print(f"\n{anomaly_type.upper()}:")
        # print(f"  - Detection rate: {detection_rate:.2f}%")
        # print(f"  - Average score: {avg_score:.6f}")
        # print(f"  - Score range: {min_score:.6f} to {max_score:.6f}")

        # Find hardest to detect examples
        if not group['is_detected'].all() and len(group) > 1:
            missed = group[~group['is_detected']].sort_values('score', ascending=True)
            # if len(missed) > 0:
            #     print(f"  - Hardest to detect example (score: {missed.iloc[0]['score']:.6f}):")
            #     for feature in ['alt', 'gs', 'heading', 'vertRate','gs_change_rate','heading_change_rate']:
            #         if feature in missed.columns:
            #             print(f"    {feature}: {missed.iloc[0][feature]}")

def flight_prediction(data, model, scaler, detector):
    """Predict anomalies in flight data using the trained model"""
    features = ['alt', 'gs', 'heading', 'vertRate', 'altChange_encoded']
    
    
    scaled = scaler.transform(data[features].values)
    anomalies = detector.predict(scaled)
    return anomalies

def main():
    try:
        # 1. Load and preprocess normal data
        print("Loading and preprocessing data...")
        df = minimal_preprocess('data/data_MVP.json')
        
        # 2. Prepare features
        features = ['alt', 'gs', 'heading', 'vertRate', 'altChange_encoded']
        # Add new engineered features if they exist
        if 'gs_change_rate' in df.columns:
            features.append('gs_change_rate')
        if 'heading_change_rate' in df.columns:
            features.append('heading_change_rate')

        X = df[features].values

        # 3. Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 4. Split into train/validation sets
        X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

        # 5. Create and train the enhanced model
        # print("\nTraining enhanced model with larger architecture...")
        model = EnhancedFlightPrediction(input_size=len(features))
        model, train_losses, val_losses = train_model(model, X_train, X_val, epochs=150, batch_size=128)
        #model, scaler, detector = load_model()
        # 6. Generate synthetic commercial flight anomalies
        # print("\nGenerating synthetic commercial flight anomalies...")
        anomaly_df = create_synthetic_anomalies(df, num_anomalies=1000)  # Increased number of anomalies

        # Create a dataframe for analysis that includes the anomaly type
        anomaly_types = anomaly_df['anomaly_type'].copy()
        # print(anomaly_types)
        # Extract only the features for anomaly detection
        anomaly_features = anomaly_df[features].values
        anomaly_features_scaled = scaler.transform(anomaly_features)


         # 6a. Combine original data with synthetic anomalies if you want a single dataset
        combined_df = pd.concat([df, anomaly_df], ignore_index=True)
        # print(f"Combined data now has {len(combined_df)} rows (normal + synthetic).")

        # 6b. (Optional) Save combined data if you want to load it in Flask or other scripts
        combined_json_path = "data/combined_MVP.json"
        combined_df.to_json(combined_json_path, orient='records')
        # print(f"Saved combined normal + anomaly data to {combined_json_path}")

        # 7. Create and evaluate anomaly detector with adjusted threshold
        # print("\nEvaluating commercial flight anomaly detection...")
        detector = AnomalyDetector(model, scaler, threshold_multiplier=2.5)  # Adjusted threshold
        detector.fit_threshold(X_train)

        # Basic evaluation
        results = detector.evaluate(X_val, anomaly_features_scaled)

        # 8. Detailed analysis by anomaly type
        # print("\nPerforming detailed analysis by anomaly type...")
        with torch.no_grad():
            anomaly_tensor = torch.FloatTensor(anomaly_features_scaled)
            predictions = model(anomaly_tensor)
            reconstruction_errors = torch.mean((anomaly_tensor - predictions) ** 2, dim=1).numpy()

        # Add the anomaly type back for analysis
        anomaly_analysis_df = pd.DataFrame({
            'anomaly_type': anomaly_types,
            'score': reconstruction_errors,
            'is_detected': reconstruction_errors > detector.threshold.item()
        })

        # Include original feature values for analysis
        for col in features:
            if col in anomaly_df.columns:
                anomaly_analysis_df[col] = anomaly_df[col].values

        analyze_anomaly_scores(anomaly_analysis_df, reconstruction_errors, detector.threshold.item())

        # 9. Save model and components
        # print("\nSaving enhanced model...")
        save_model(model, scaler, detector)

        # Optionally load the model back to verify
        #

        
        
        # print("\nEnhanced model training and evaluation complete!")




    except Exception as e:
        # print(f"Error in main execution: {str(e)}")
        raise
    
# def main2():
#     test_df = pd.DataFrame(
#         {
#             "id": "QTR176-1681826580-schedule-0777",
#             "pitr": 1682015957.0,
#             "alt": 45200.0,
#             "altChange": " ",
#             "gs": 495.0,
#             "heading": 142.0,
#             "lat": 31.91217,
#             "lon": 46.53798,
#             "vertRate": 0.0,
#             "anomaly": 0
#         },
# )
    
#     model, scaler, detector = load_model()
#     features = ['alt', 'gs', 'heading', 'lat', 'lon', 'vertRate', 'altChange_encoded']
#     scaled = scaler.transform(test_df[features].values)
#     model.predict(scaled)
#     print(test_df)
    


if __name__ == "__main__":
    main()