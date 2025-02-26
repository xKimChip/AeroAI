import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import json

def prepare_data(json_data):
    data = pd.DataFrame(json_data)
    
    features = [
        'alt',           # altitude
        'gs',           # ground speed
        'speed_ias',    # indicated airspeed
        'speed_tas',    # true airspeed
        'mach',         # mach number
        'temperature',  # temperature
        'pressure',     # pressure
        'vertRate',     # vertical rate
        'wind_speed'    # wind speed
    ]
    
    for feature in features:
        if feature in data.columns:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
    
    data_cleaned = data[features].dropna()
    return data_cleaned.reset_index(drop=True)

def create_extreme_aviation_anomaly(base_flight, anomaly_type, data_stats):
    """Create more extreme aviation-specific anomalies"""
    anomaly = base_flight.copy()
    
    if anomaly_type == 'speed_violation':
        # Create physically impossible speed relationships
        anomaly['speed_tas'] = 800  # Extreme speed for commercial aircraft
        anomaly['gs'] = 900        # Ground speed > true airspeed (impossible)
        anomaly['speed_ias'] = 400 # Very high IAS
        anomaly['mach'] = 2.0      # Supersonic
        
    elif anomaly_type == 'altitude_pressure':
        # Create impossible altitude-pressure relationships
        anomaly['alt'] = 45000     # Above typical service ceiling
        anomaly['pressure'] = 1013  # Sea level pressure at high altitude
        anomaly['temperature'] = 30 # Too warm for high altitude
        anomaly['mach'] = 0.2      # Too slow for high altitude
        
    elif anomaly_type == 'vertical_performance':
        # Create impossible climb/descent scenarios
        anomaly['vertRate'] = 10000  # Extreme vertical rate
        anomaly['speed_ias'] = 150   # Too slow for normal flight
        anomaly['alt'] = 38000       # High altitude
        anomaly['mach'] = 0.3        # Too slow for altitude
        
    elif anomaly_type == 'environmental':
        # Create impossible environmental conditions
        anomaly['wind_speed'] = data_stats['wind_speed']['mean'] + (data_stats['wind_speed']['std'] * 10)
        anomaly['temperature'] = -100  # Extreme cold
        anomaly['pressure'] = data_stats['pressure']['mean'] - (data_stats['pressure']['std'] * 8)
        
    elif anomaly_type == 'combined':
        # Multiple impossible conditions
        anomaly['alt'] = 42000
        anomaly['speed_tas'] = 750
        anomaly['speed_ias'] = 50    # Impossible IAS/TAS relationship
        anomaly['vertRate'] = 8000
        anomaly['wind_speed'] = 300
        anomaly['temperature'] = -90
        
    return anomaly

def generate_aviation_anomalies(data, num_anomalies=5):
    """Generate more extreme aviation-specific anomalies"""
    anomalies = []
    
    # Calculate statistics for each feature
    data_stats = {col: {'mean': data[col].mean(), 'std': data[col].std()} 
                 for col in data.columns}
    
    anomaly_types = [
        'speed_violation',
        'altitude_pressure',
        'vertical_performance',
        'environmental',
        'combined'
    ]
    
    for anomaly_type in anomaly_types:
        base_flight = data.sample(n=1).iloc[0]
        anomaly = create_extreme_aviation_anomaly(base_flight, anomaly_type, data_stats)
        anomalies.append(anomaly)
    
    anomalies_df = pd.DataFrame(anomalies).reset_index(drop=True)
    combined_data = pd.concat([data, anomalies_df], ignore_index=True)
    
    labels = np.zeros(len(combined_data))
    labels[len(data):] = 1
    
    return combined_data, labels, anomalies_df

def train_and_evaluate_isolation_forest(data, labels, random_state=42, param_set='default'):
    """Train and evaluate isolation forest with different parameter sets"""
    parameter_sets = {
        'default': {
            'contamination': 0.02,
            'max_features': 0.7,
            'n_estimators': 200,
        },
        'aggressive': {
            'contamination': 0.05,  # More aggressive anomaly detection
            'max_features': 0.5,    # Consider fewer features per split
            'n_estimators': 300,    # More trees
        },
        'conservative': {
            'contamination': 0.01,  # More conservative
            'max_features': 0.9,    # Consider most features
            'n_estimators': 100,    # Fewer trees
        }
    }
    
    params = parameter_sets[param_set]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Calculate appropriate contamination based on synthetic anomalies
    contamination = len(labels[labels == 1]) / len(labels)
    
    # Try a higher contamination rate to be more sensitive to anomalies
    # and specify max_features to consider subset of features for each split
    iso_forest = IsolationForest(
        contamination=0.02,  # Allow for 2% anomalies instead of exact ratio
        random_state=random_state,
        n_estimators=200,
        max_samples='auto',
        max_features=0.7,    # Use 70% of features for each tree
        bootstrap=True
    )
    
    print(f"\nModel Parameters:")
    print(f"Contamination rate: 0.02 (2% of points may be anomalies)")
    print(f"Max features per tree: 0.7 (70% of features)")
    print(f"Number of estimators: 200")
    
    # Get anomaly scores
    scores = iso_forest.fit(data_scaled).score_samples(data_scaled)
    
    # Use score threshold instead of contamination-based prediction
    threshold = np.percentile(scores, 5)  # Use bottom 5% as threshold
    predictions = np.where(scores < threshold, -1, 1)
    
    # Convert to binary format for metrics
    predictions_binary = [1 if pred == -1 else 0 for pred in predictions]
    
    print(f"\nScore Statistics:")
    print(f"Average score: {np.mean(scores):.3f}")
    print(f"Score threshold: {threshold:.3f}")
    print(f"Minimum score: {np.min(scores):.3f}")
    print(f"Maximum score: {np.max(scores):.3f}")
    print("\nSynthetic Anomaly Scores:")
    
    metrics = {
        'precision': precision_score(labels, predictions_binary),
        'recall': recall_score(labels, predictions_binary),
        'f1_score': f1_score(labels, predictions_binary)
    }
    
    return predictions, scores, metrics

def analyze_anomalies(data, predictions, scores, labels, anomalies_df):
    results = pd.DataFrame({
        'anomaly_predicted': predictions,
        'anomaly_actual': labels,
        'anomaly_score': scores
    })
    
    for col in data.columns:
        results[col] = data[col].values
    
    # Calculate feature importance
    feature_importance = {}
    normal_data = data.iloc[:-len(anomalies_df)]
    
    for feature in data.columns:
        normal_mean = normal_data[feature].mean()
        normal_std = normal_data[feature].std()
        anomaly_values = anomalies_df[feature]
        z_scores = abs((anomaly_values - normal_mean) / normal_std)
        feature_importance[feature] = z_scores.mean()
    
    # Calculate detection statistics
    detection_stats = {
        'total_predictions': len(predictions),
        'predicted_anomalies': sum(predictions == -1),
        'actual_anomalies': sum(labels == 1),
        'true_positives': sum((predictions == -1) & (labels == 1)),
        'false_positives': sum((predictions == -1) & (labels == 0)),
        'false_negatives': sum((predictions == 1) & (labels == 1))
    }
    
    return results, feature_importance, detection_stats

def main():
    # Load data
    with open('flight_data.json', 'r') as f:
        json_data = json.load(f)
    
    # Prepare data
    data_cleaned = prepare_data(json_data)
    
    # Generate anomalies
    data_with_anomalies, labels, anomalies_df = generate_aviation_anomalies(data_cleaned)
    
    # Try different parameter sets
    parameter_sets = ['default', 'aggressive', 'conservative']
    
    print("\nTrying different parameter sets:")
    best_metrics = None
    best_results = None
    best_param_set = None
    
    for param_set in parameter_sets:
        print(f"\nTesting {param_set} parameters:")
        predictions, scores, metrics = train_and_evaluate_isolation_forest(
            data_with_anomalies, 
            labels,
            param_set=param_set
        )
        
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        
        # Keep track of best performing parameters
        if best_metrics is None or metrics['f1_score'] > best_metrics['f1_score']:
            best_metrics = metrics
            best_results = (predictions, scores)
            best_param_set = param_set
    
    print(f"\nBest performing parameter set: {best_param_set}")
    predictions, scores = best_results
    
    # Analyze results
    results, feature_importance, detection_stats = analyze_anomalies(
        data_with_anomalies, 
        predictions, 
        scores, 
        labels, 
        anomalies_df
    )
    
    # Print comprehensive results
    print("\nModel Performance Metrics:")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    
    print("\nDetection Statistics:")
    print(f"Total data points: {detection_stats['total_predictions']}")
    print(f"Predicted anomalies: {detection_stats['predicted_anomalies']}")
    print(f"Actual anomalies: {detection_stats['actual_anomalies']}")
    print(f"True positives: {detection_stats['true_positives']}")
    print(f"False positives: {detection_stats['false_positives']}")
    print(f"False negatives: {detection_stats['false_negatives']}")
    
    print("\nFeature Importance in Anomaly Detection:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.2f} standard deviations")
    
    print("\nSynthetic Anomalies Generated:")
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(anomalies_df)
    
    print("\nCorrectly Detected Anomalies (True Positives):")
    true_positives = results[
        (results['anomaly_predicted'] == -1) & 
        (results['anomaly_actual'] == 1)
    ]
    if not true_positives.empty:
        print(true_positives[['anomaly_score'] + list(data_cleaned.columns)])
    else:
        print("No anomalies were correctly detected!")
    
    print("\nMissed Anomalies (False Negatives):")
    false_negatives = results[
        (results['anomaly_predicted'] == 1) & 
        (results['anomaly_actual'] == 1)
    ]
    if not false_negatives.empty:
        print(false_negatives[['anomaly_score'] + list(data_cleaned.columns)])
    else:
        print("No anomalies were missed!")

if __name__ == "__main__":
    main()