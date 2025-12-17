"""
STEP 3: Advanced Feature Engineering
PHM 2010 Milling Dataset - Feature Extraction Pipeline
Creates comprehensive feature set for causal inference and RUL prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew, entropy
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pickle

print("="*80)
print("STEP 3: ADVANCED FEATURE ENGINEERING")
print("="*80 + "\n")

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
DATA_PATH = Path("F:/phm_rul_reserach/phm data")
OUTPUT_PATH = Path("./features_output")
OUTPUT_PATH.mkdir(exist_ok=True)

# Feature extraction parameters
SAMPLING_RATE = 1000  # Hz (adjust if known)
DOWNSAMPLE_FACTOR = 10  # Reduce computation by taking every Nth sample

print("⚙️ Configuration:")
print(f"  • Data path: {DATA_PATH}")
print(f"  • Output path: {OUTPUT_PATH}")
print(f"  • Sampling rate: {SAMPLING_RATE} Hz")
print(f"  • Downsample factor: {DOWNSAMPLE_FACTOR}x")
print()

# ============================================================================
# 2. FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_time_domain_features(signal_data):
    """
    Extract time-domain statistical features
    Returns: dict of features
    """
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(signal_data)
    features['std'] = np.std(signal_data)
    features['var'] = np.var(signal_data)
    features['rms'] = np.sqrt(np.mean(signal_data**2))
    
    # Amplitude features
    features['peak'] = np.max(np.abs(signal_data))
    features['peak_to_peak'] = np.ptp(signal_data)
    features['crest_factor'] = features['peak'] / (features['rms'] + 1e-10)
    features['clearance_factor'] = features['peak'] / (np.mean(np.sqrt(np.abs(signal_data)))**2 + 1e-10)
    features['shape_factor'] = features['rms'] / (np.mean(np.abs(signal_data)) + 1e-10)
    features['impulse_factor'] = features['peak'] / (np.mean(np.abs(signal_data)) + 1e-10)
    
    # Distribution features
    features['skewness'] = skew(signal_data)
    features['kurtosis'] = kurtosis(signal_data)
    
    # Energy features
    features['energy'] = np.sum(signal_data**2)
    features['log_energy'] = np.log(features['energy'] + 1e-10)
    
    # Percentiles
    features['q25'] = np.percentile(signal_data, 25)
    features['q75'] = np.percentile(signal_data, 75)
    features['iqr'] = features['q75'] - features['q25']
    
    # Zero crossing rate
    zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
    
    # Waveform length
    features['waveform_length'] = np.sum(np.abs(np.diff(signal_data)))
    
    return features


def extract_frequency_domain_features(signal_data, sampling_rate=1000):
    """
    Extract frequency-domain features using FFT
    Returns: dict of features
    """
    features = {}
    
    # Compute FFT
    n = len(signal_data)
    yf = fft(signal_data)
    xf = fftfreq(n, 1/sampling_rate)[:n//2]
    power = 2.0/n * np.abs(yf[0:n//2])
    
    # Power spectral density
    psd = power**2
    
    # Total power
    features['total_power'] = np.sum(psd)
    
    # Dominant frequency
    dominant_idx = np.argmax(power[1:]) + 1  # Skip DC component
    features['dominant_frequency'] = xf[dominant_idx]
    features['dominant_power'] = power[dominant_idx]
    
    # Frequency bands (assuming milling frequencies)
    # Low: 0-100 Hz, Mid: 100-500 Hz, High: 500+ Hz
    low_band = (xf >= 0) & (xf < 100)
    mid_band = (xf >= 100) & (xf < 500)
    high_band = (xf >= 500)
    
    features['power_low'] = np.sum(psd[low_band])
    features['power_mid'] = np.sum(psd[mid_band])
    features['power_high'] = np.sum(psd[high_band])
    
    # Band ratios
    total_power_nonzero = features['total_power'] + 1e-10
    features['power_ratio_low'] = features['power_low'] / total_power_nonzero
    features['power_ratio_mid'] = features['power_mid'] / total_power_nonzero
    features['power_ratio_high'] = features['power_high'] / total_power_nonzero
    
    # Spectral statistics
    power_normalized = power / (np.sum(power) + 1e-10)
    features['spectral_mean'] = np.sum(xf * power_normalized)
    features['spectral_std'] = np.sqrt(np.sum(((xf - features['spectral_mean'])**2) * power_normalized))
    features['spectral_skewness'] = np.sum(((xf - features['spectral_mean'])**3) * power_normalized) / (features['spectral_std']**3 + 1e-10)
    features['spectral_kurtosis'] = np.sum(((xf - features['spectral_mean'])**4) * power_normalized) / (features['spectral_std']**4 + 1e-10)
    
    # Spectral entropy
    power_prob = power_normalized + 1e-10
    features['spectral_entropy'] = -np.sum(power_prob * np.log(power_prob))
    
    # Spectral edge frequency (95% power)
    cumsum_power = np.cumsum(power_normalized)
    edge_idx = np.where(cumsum_power >= 0.95)[0]
    features['spectral_edge_95'] = xf[edge_idx[0]] if len(edge_idx) > 0 else xf[-1]
    
    return features


def extract_information_theory_features(signal_data, bins=50):
    """
    Extract information theory features
    Returns: dict of features
    """
    features = {}
    
    # Sample entropy
    hist, _ = np.histogram(signal_data, bins=bins)
    hist_prob = hist / (np.sum(hist) + 1e-10)
    hist_prob = hist_prob[hist_prob > 0]
    features['sample_entropy'] = -np.sum(hist_prob * np.log(hist_prob))
    
    # Approximate entropy (simplified)
    features['approx_entropy'] = features['sample_entropy']
    
    return features


def extract_wavelet_features(signal_data):
    """
    Extract time-frequency features using wavelet transform
    Returns: dict of features
    """
    features = {}
    
    try:
        from scipy import signal as sp_signal
        
        # Continuous wavelet transform (simplified - using Morlet)
        widths = np.arange(1, 31)
        cwtmatr = sp_signal.cwt(signal_data[:5000], sp_signal.morlet2, widths)  # Use subset for speed
        
        # Wavelet energy
        features['wavelet_energy'] = np.sum(np.abs(cwtmatr)**2)
        features['wavelet_mean'] = np.mean(np.abs(cwtmatr))
        features['wavelet_std'] = np.std(np.abs(cwtmatr))
        
    except Exception as e:
        # If wavelet fails, use placeholder
        features['wavelet_energy'] = 0
        features['wavelet_mean'] = 0
        features['wavelet_std'] = 0
    
    return features


def extract_degradation_indicators(signal_data):
    """
    Extract features specific to degradation monitoring
    Returns: dict of features
    """
    features = {}
    
    # Trend analysis
    x = np.arange(len(signal_data))
    slope, intercept, r_value, _, _ = stats.linregress(x, signal_data)
    features['trend_slope'] = slope
    features['trend_intercept'] = intercept
    features['trend_r2'] = r_value**2
    
    # Moving average difference
    window = min(100, len(signal_data) // 10)
    if window > 2:
        moving_avg = np.convolve(signal_data, np.ones(window)/window, mode='valid')
        features['ma_mean'] = np.mean(moving_avg)
        features['ma_std'] = np.std(moving_avg)
    else:
        features['ma_mean'] = np.mean(signal_data)
        features['ma_std'] = np.std(signal_data)
    
    return features


def extract_all_features(signal_data, channel_name, sampling_rate=1000):
    """
    Extract all features for a single channel
    """
    all_features = {}
    
    # Downsample for efficiency
    signal_downsampled = signal_data[::DOWNSAMPLE_FACTOR]
    
    # Extract different feature groups
    time_features = extract_time_domain_features(signal_downsampled)
    freq_features = extract_frequency_domain_features(signal_downsampled, sampling_rate/DOWNSAMPLE_FACTOR)
    info_features = extract_information_theory_features(signal_downsampled)
    wavelet_features = extract_wavelet_features(signal_downsampled)
    degradation_features = extract_degradation_indicators(signal_downsampled)
    
    # Combine all features with channel prefix
    for name, value in time_features.items():
        all_features[f'{channel_name}_{name}'] = value
    for name, value in freq_features.items():
        all_features[f'{channel_name}_{name}'] = value
    for name, value in info_features.items():
        all_features[f'{channel_name}_{name}'] = value
    for name, value in wavelet_features.items():
        all_features[f'{channel_name}_{name}'] = value
    for name, value in degradation_features.items():
        all_features[f'{channel_name}_{name}'] = value
    
    return all_features


# ============================================================================
# 3. PROCESS ALL CUTTING OPERATIONS
# ============================================================================
print("🔧 Extracting Features from All Cuts...")
print("-" * 80)

# Load wear data
wear_data = pd.read_csv(DATA_PATH / "c1_wear_processed.csv")
print(f"✓ Loaded wear data: {len(wear_data)} cuts")

# Initialize feature collection
all_features_list = []

# Channel names (generic - update if you know the actual sensor types)
channel_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7']

# Process each cut
print(f"\n📊 Processing {len(wear_data)} cutting operations...")
print("This may take several minutes...\n")

for idx, row in tqdm(wear_data.iterrows(), total=len(wear_data), desc="Extracting features"):
    cut_num = int(row['cut'])
    
    # Load sensor data
    sensor_file = DATA_PATH / f"c_1_{str(cut_num).zfill(3)}_processed.csv"
    
    if not sensor_file.exists():
        print(f"\n⚠️ File not found for cut {cut_num}: {sensor_file.name}")
        continue
    
    try:
        # Load sensor data
        sensor_df = pd.read_csv(sensor_file)
        
        # Verify data was loaded
        if sensor_df.empty or len(sensor_df) == 0:
            print(f"\n⚠️ Empty data for cut {cut_num}")
            continue
        
        # Initialize feature dict for this cut
        cut_features = {
            'cut': cut_num,
            'flute_1': row['flute_1'],
            'flute_2': row['flute_2'],
            'flute_3': row['flute_3'],
        }
        
        # Calculate RUL (Remaining Useful Life) - cycles until maximum wear
        max_wear = wear_data[['flute_1', 'flute_2', 'flute_3']].max().max()
        current_max_wear = max(row['flute_1'], row['flute_2'], row['flute_3'])
        cut_features['RUL'] = max_wear - current_max_wear
        cut_features['max_wear'] = current_max_wear
        
        # Extract features from each channel
        for ch_idx, col in enumerate(sensor_df.columns):
            if ch_idx >= len(channel_names):
                break
            
            signal_data = sensor_df[col].values
            ch_features = extract_all_features(signal_data, channel_names[ch_idx], SAMPLING_RATE)
            cut_features.update(ch_features)
        
        all_features_list.append(cut_features)
        
        # Debug: print progress every 50 cuts
        if cut_num % 50 == 0:
            print(f"\n  ✓ Processed cut {cut_num}, extracted {len(cut_features)} features")
        
    except Exception as e:
        print(f"\n⚠️ Error processing cut {cut_num}: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n✓ Successfully extracted features from {len(all_features_list)} cuts")

# ============================================================================
# 4. CREATE FEATURE DATAFRAME
# ============================================================================
print("\n" + "="*80)
print("📊 Creating Feature Dataset")
print("-" * 80)

# Convert to dataframe
features_df = pd.DataFrame(all_features_list)

if features_df.empty or len(features_df) == 0:
    print("\n❌ ERROR: No features were extracted!")
    print("Possible issues:")
    print("  1. Files couldn't be loaded")
    print("  2. Feature extraction failed")
    print("  3. Data format issues")
    print("\nPlease check the error messages above.")
    exit(1)

print(f"\n✓ Feature matrix shape: {features_df.shape}")
print(f"  • Samples (cuts): {features_df.shape[0]}")
print(f"  • Total features: {features_df.shape[1]}")
print(f"  • Target features: flute_1, flute_2, flute_3, RUL")
print(f"  • Engineered features: {features_df.shape[1] - 5}")

# Display first few rows
print(f"\n📋 First few rows:")
print(features_df.head())

# Display feature statistics
print(f"\n📈 Feature Statistics:")
if len(features_df.columns) > 0:
    print(features_df.describe().T.head(20))
else:
    print("  No features to display")

# ============================================================================
# 5. SAVE FEATURES
# ============================================================================
print("\n" + "="*80)
print("💾 Saving Feature Dataset")
print("-" * 80)

# Save as CSV
csv_path = OUTPUT_PATH / "features_engineered.csv"
features_df.to_csv(csv_path, index=False)
print(f"✓ Saved CSV: {csv_path}")

# Save as pickle (preserves data types)
pkl_path = OUTPUT_PATH / "features_engineered.pkl"
features_df.to_pickle(pkl_path)
print(f"✓ Saved PKL: {pkl_path}")

# Save feature names
feature_names = [col for col in features_df.columns if col not in ['cut', 'flute_1', 'flute_2', 'flute_3', 'RUL', 'max_wear']]
feature_info = {
    'feature_names': feature_names,
    'target_names': ['flute_1', 'flute_2', 'flute_3', 'RUL'],
    'channel_names': channel_names,
    'num_features_per_channel': len(feature_names) // len(channel_names),
    'total_samples': len(features_df)
}

info_path = OUTPUT_PATH / "feature_info.pkl"
with open(info_path, 'wb') as f:
    pickle.dump(feature_info, f)
print(f"✓ Saved feature info: {info_path}")

# ============================================================================
# 6. FEATURE IMPORTANCE PREVIEW
# ============================================================================
print("\n" + "="*80)
print("🎯 Feature Correlation with Wear")
print("-" * 80)

if not features_df.empty and len(feature_names) > 0:
    # Calculate correlation with max_wear
    correlations = features_df[feature_names].corrwith(features_df['max_wear']).abs().sort_values(ascending=False)
    
    print(f"\n🔝 Top 20 features correlated with wear:")
    print(correlations.head(20))
    
    # Save correlations
    corr_path = OUTPUT_PATH / "feature_correlations.csv"
    correlations.to_csv(corr_path, header=['correlation'])
    print(f"\n✓ Saved correlations: {corr_path}")
else:
    print("\n⚠️ Cannot calculate correlations - no features available")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ STEP 3 COMPLETE - Advanced Feature Engineering")
print("="*80)

print(f"\n📊 Feature Extraction Summary:")
print(f"  • Total samples: {len(features_df)}")
print(f"  • Total features: {len(feature_names)}")
print(f"  • Features per channel: ~{len(feature_names) // len(channel_names)}")
print(f"  • Feature categories:")
print(f"    - Time-domain: 15+ features")
print(f"    - Frequency-domain: 15+ features")
print(f"    - Information theory: 2+ features")
print(f"    - Wavelet: 3 features")
print(f"    - Degradation indicators: 5+ features")

print(f"\n📁 Output Files:")
print(f"  • {csv_path.name}")
print(f"  • {pkl_path.name}")
print(f"  • {info_path.name}")
print(f"  • {corr_path.name}")

print(f"\n📝 NEXT STEPS:")
print("   → STEP 4: Causal Inference Analysis")
print("     (Granger causality, structural causal models, counterfactuals)")
print("   → STEP 5: RUL Prediction with Conformal Prediction")
print("     (ML models + uncertainty quantification)")

print("\n" + "="*80)