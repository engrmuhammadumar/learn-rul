"""
STEP 2: Comprehensive Exploratory Data Analysis (EDA)
PHM 2010 Milling Dataset - Advanced RUL Research
Focus: Understanding wear patterns, sensor correlations, and degradation dynamics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*80)
print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80 + "\n")

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
DATA_PATH = Path("F:/phm_rul_reserach/phm data")
OUTPUT_PATH = Path("./eda_outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

# ============================================================================
# 2. LOAD DATA
# ============================================================================
print("📊 Loading Data...")
print("-" * 80)

# Load wear data
wear_data = pd.read_csv(DATA_PATH / "c1_wear_processed.csv")
print(f"✓ Wear data loaded: {wear_data.shape}")

# Load sample sensor data for analysis (first 10 operations)
sensor_files = [f"c_1_{str(i).zfill(3)}_processed.csv" for i in range(1, 11)]
sensor_data_dict = {}

for file in sensor_files:
    file_path = DATA_PATH / file
    if file_path.exists():
        df = pd.read_csv(file_path)
        cut_num = int(file.split('_')[2])
        sensor_data_dict[cut_num] = df

print(f"✓ Loaded {len(sensor_data_dict)} sensor data files for EDA")
print()

# ============================================================================
# 3. WEAR PROGRESSION ANALYSIS
# ============================================================================
print("📈 ANALYSIS 3.1: Tool Wear Progression")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Tool Wear Progression Analysis', fontsize=16, fontweight='bold')

# 3.1: Wear progression over cuts
ax = axes[0, 0]
ax.plot(wear_data['cut'], wear_data['flute_1'], 'o-', label='Flute 1', alpha=0.7, linewidth=2)
ax.plot(wear_data['cut'], wear_data['flute_2'], 's-', label='Flute 2', alpha=0.7, linewidth=2)
ax.plot(wear_data['cut'], wear_data['flute_3'], '^-', label='Flute 3', alpha=0.7, linewidth=2)
ax.set_xlabel('Cut Number', fontsize=11, fontweight='bold')
ax.set_ylabel('Wear (μm)', fontsize=11, fontweight='bold')
ax.set_title('A. Wear Progression Over Time', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Calculate wear rate
wear_data['wear_rate_f1'] = wear_data['flute_1'].diff()
wear_data['wear_rate_f2'] = wear_data['flute_2'].diff()
wear_data['wear_rate_f3'] = wear_data['flute_3'].diff()

print(f"Initial wear (cut 1):")
print(f"  Flute 1: {wear_data['flute_1'].iloc[0]:.2f} μm")
print(f"  Flute 2: {wear_data['flute_2'].iloc[0]:.2f} μm")
print(f"  Flute 3: {wear_data['flute_3'].iloc[0]:.2f} μm")

print(f"\nFinal wear (cut 315):")
print(f"  Flute 1: {wear_data['flute_1'].iloc[-1]:.2f} μm")
print(f"  Flute 2: {wear_data['flute_2'].iloc[-1]:.2f} μm")
print(f"  Flute 3: {wear_data['flute_3'].iloc[-1]:.2f} μm")

print(f"\nAverage wear rate (μm/cut):")
print(f"  Flute 1: {wear_data['wear_rate_f1'].mean():.3f}")
print(f"  Flute 2: {wear_data['wear_rate_f2'].mean():.3f}")
print(f"  Flute 3: {wear_data['wear_rate_f3'].mean():.3f}")

# 3.2: Wear rate progression
ax = axes[0, 1]
ax.plot(wear_data['cut'][1:], wear_data['wear_rate_f1'][1:], 'o-', 
        label='Flute 1 Rate', alpha=0.6, markersize=3)
ax.plot(wear_data['cut'][1:], wear_data['wear_rate_f2'][1:], 's-', 
        label='Flute 2 Rate', alpha=0.6, markersize=3)
ax.plot(wear_data['cut'][1:], wear_data['wear_rate_f3'][1:], '^-', 
        label='Flute 3 Rate', alpha=0.6, markersize=3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Cut Number', fontsize=11, fontweight='bold')
ax.set_ylabel('Wear Rate (μm/cut)', fontsize=11, fontweight='bold')
ax.set_title('B. Wear Rate Progression', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# 3.3: Distribution of wear values
ax = axes[1, 0]
ax.hist(wear_data['flute_1'], bins=30, alpha=0.6, label='Flute 1', edgecolor='black')
ax.hist(wear_data['flute_2'], bins=30, alpha=0.6, label='Flute 2', edgecolor='black')
ax.hist(wear_data['flute_3'], bins=30, alpha=0.6, label='Flute 3', edgecolor='black')
ax.set_xlabel('Wear (μm)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('C. Wear Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 3.4: Correlation between flutes
ax = axes[1, 1]
wear_corr = wear_data[['flute_1', 'flute_2', 'flute_3']].corr()
im = ax.imshow(wear_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(['Flute 1', 'Flute 2', 'Flute 3'], fontsize=10)
ax.set_yticklabels(['Flute 1', 'Flute 2', 'Flute 3'], fontsize=10)
ax.set_title('D. Inter-Flute Correlation', fontsize=12, fontweight='bold')

# Add correlation values
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, f'{wear_corr.iloc[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=11)

plt.colorbar(im, ax=ax, label='Correlation')
plt.tight_layout()
plt.savefig(OUTPUT_PATH / '01_wear_progression.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: 01_wear_progression.png")
plt.close()

print(f"\n📊 Inter-flute correlations:")
print(wear_corr)

# ============================================================================
# 4. SENSOR SIGNAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("🔬 ANALYSIS 4.1: Sensor Signal Characteristics")
print("-" * 80)

# Analyze first few cuts (early wear stage)
early_cuts = [1, 2, 3]
# Analyze middle cuts (medium wear stage)
mid_cuts = [157, 158, 159]
# Analyze late cuts (high wear stage)
late_cuts = [313, 314, 315]

stages = {
    'Early (Low Wear)': early_cuts,
    'Middle (Medium Wear)': mid_cuts,
    'Late (High Wear)': late_cuts
}

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('Sensor Signal Analysis Across Wear Stages', fontsize=16, fontweight='bold')

sensor_names = ['Force X', 'Force Y', 'Force Z', 'Vib X', 'Vib Y', 'Vib Z', 'AE']

for stage_idx, (stage_name, cuts) in enumerate(stages.items()):
    # Get first cut from this stage
    cut_num = cuts[0]
    
    if cut_num in sensor_data_dict:
        df = sensor_data_dict[cut_num]
        
        # Time series plot (first 3 channels)
        ax = axes[stage_idx, 0]
        time = np.arange(len(df)) / 1000  # Convert to seconds (assuming kHz sampling)
        
        for i in range(min(3, len(df.columns))):
            ax.plot(time[:5000], df.iloc[:5000, i], alpha=0.7, linewidth=0.8, 
                   label=f'Ch{i+1}')
        
        ax.set_xlabel('Time (s)' if stage_idx == 2 else '', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
        ax.set_title(f'{stage_name} - Time Series', fontsize=11, fontweight='bold')
        if stage_idx == 0:
            ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Statistical summary
        ax = axes[stage_idx, 1]
        channel_stats = []
        for col in df.columns:
            channel_stats.append([
                df[col].mean(),
                df[col].std(),
                df[col].max() - df[col].min()
            ])
        
        channel_stats = np.array(channel_stats)
        x_pos = np.arange(len(df.columns))
        width = 0.25
        
        ax.bar(x_pos - width, channel_stats[:, 0], width, label='Mean', alpha=0.7)
        ax.bar(x_pos, channel_stats[:, 1], width, label='Std Dev', alpha=0.7)
        ax.bar(x_pos + width, channel_stats[:, 2]/10, width, label='Range/10', alpha=0.7)
        
        ax.set_xlabel('Channel' if stage_idx == 2 else '', fontsize=10)
        ax.set_ylabel('Value', fontsize=10, fontweight='bold')
        ax.set_title(f'{stage_name} - Statistics', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Ch{i+1}' for i in range(len(df.columns))], fontsize=8)
        if stage_idx == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Frequency domain (first channel)
        ax = axes[stage_idx, 2]
        signal_data = df.iloc[:, 0].values
        n = len(signal_data)
        
        # Compute FFT
        yf = fft(signal_data)
        xf = fftfreq(n, 1.0)[:n//2]
        
        # Plot power spectrum
        power = 2.0/n * np.abs(yf[0:n//2])
        ax.semilogy(xf[:1000], power[:1000], linewidth=1.5)
        ax.set_xlabel('Frequency (Hz)' if stage_idx == 2 else '', fontsize=10)
        ax.set_ylabel('Power', fontsize=10, fontweight='bold')
        ax.set_title(f'{stage_name} - Frequency Spectrum', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(power[1:1000]) + 1
        dominant_freq = xf[dominant_freq_idx]
        ax.axvline(x=dominant_freq, color='r', linestyle='--', alpha=0.5, 
                  label=f'Peak: {dominant_freq:.2f} Hz')
        if stage_idx == 0:
            ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / '02_sensor_signal_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 02_sensor_signal_analysis.png")
plt.close()

# ============================================================================
# 5. FEATURE STATISTICS ACROSS WEAR STAGES
# ============================================================================
print("\n" + "="*80)
print("📊 ANALYSIS 5.1: Feature Evolution Across Wear Stages")
print("-" * 80)

# Compute features for different wear stages
feature_stats = []

for stage_name, cuts in stages.items():
    stage_features = []
    
    for cut_num in cuts:
        if cut_num in sensor_data_dict:
            df = sensor_data_dict[cut_num]
            
            # Compute RMS for each channel
            rms_values = np.sqrt(np.mean(df**2, axis=0))
            stage_features.append(rms_values)
    
    if stage_features:
        avg_features = np.mean(stage_features, axis=0)
        feature_stats.append({
            'stage': stage_name,
            'features': avg_features,
            'wear': wear_data.loc[wear_data['cut'] == cuts[0], 
                                 ['flute_1', 'flute_2', 'flute_3']].values[0]
        })

# Visualize feature evolution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Feature Evolution Across Wear Stages', fontsize=16, fontweight='bold')

# RMS evolution
ax = axes[0]
stage_names = [fs['stage'] for fs in feature_stats]
colors = ['green', 'orange', 'red']

for ch_idx in range(min(7, len(feature_stats[0]['features']))):
    rms_values = [fs['features'][ch_idx] for fs in feature_stats]
    ax.plot(stage_names, rms_values, 'o-', label=f'Channel {ch_idx+1}', 
           linewidth=2, markersize=8, alpha=0.7)

ax.set_xlabel('Wear Stage', fontsize=12, fontweight='bold')
ax.set_ylabel('RMS Value', fontsize=12, fontweight='bold')
ax.set_title('A. RMS Feature Evolution', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

# Wear vs Features
ax = axes[1]
avg_wear = [np.mean(fs['wear']) for fs in feature_stats]
avg_rms = [np.mean(fs['features']) for fs in feature_stats]

for i, (stage, wear, rms) in enumerate(zip(stage_names, avg_wear, avg_rms)):
    ax.scatter(wear, rms, s=200, alpha=0.7, color=colors[i], 
              edgecolors='black', linewidth=2, label=stage)
    ax.annotate(stage, (wear, rms), xytext=(5, 5), 
               textcoords='offset points', fontsize=10)

ax.set_xlabel('Average Wear (μm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Average RMS', fontsize=12, fontweight='bold')
ax.set_title('B. Wear vs. Signal Strength', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / '03_feature_evolution.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 03_feature_evolution.png")
plt.close()

# Print statistics
print(f"\nFeature Statistics Across Wear Stages:")
print("-" * 80)
for fs in feature_stats:
    print(f"\n{fs['stage']}:")
    print(f"  Average Wear: {np.mean(fs['wear']):.2f} μm")
    print(f"  Average RMS: {np.mean(fs['features']):.4f}")
    print(f"  Feature Range: [{np.min(fs['features']):.4f}, {np.max(fs['features']):.4f}]")

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("🔗 ANALYSIS 6.1: Sensor Channel Correlations")
print("-" * 80)

# Analyze correlations for different wear stages
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Sensor Channel Correlations Across Wear Stages', 
            fontsize=16, fontweight='bold')

for idx, (stage_name, cuts) in enumerate(stages.items()):
    cut_num = cuts[0]
    
    if cut_num in sensor_data_dict:
        df = sensor_data_dict[cut_num]
        
        # Compute correlation matrix
        corr_matrix = df.corr()
        
        # Plot
        ax = axes[idx]
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.columns)))
        ax.set_xticklabels([f'Ch{i+1}' for i in range(len(df.columns))], fontsize=9)
        ax.set_yticklabels([f'Ch{i+1}' for i in range(len(df.columns))], fontsize=9)
        ax.set_title(f'{stage_name}\n(Cut {cut_num})', fontsize=12, fontweight='bold')
        
        # Add correlation values
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", 
                             color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                             fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Correlation', fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / '04_channel_correlations.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 04_channel_correlations.png")
plt.close()

# ============================================================================
# 7. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("📋 SUMMARY STATISTICS")
print("="*80)

summary_stats = {
    'Dataset': 'PHM 2010 Milling',
    'Total Cuts': len(wear_data),
    'Sensor Channels': len(sensor_data_dict[1].columns) if 1 in sensor_data_dict else 'N/A',
    'Avg Samples per Cut': np.mean([len(sensor_data_dict[k]) for k in sensor_data_dict.keys()]),
    'Wear Range': f"[{wear_data[['flute_1', 'flute_2', 'flute_3']].min().min():.2f}, "
                  f"{wear_data[['flute_1', 'flute_2', 'flute_3']].max().max():.2f}] μm",
    'Flute Correlation': f"{wear_corr.values[np.triu_indices_from(wear_corr.values, k=1)].mean():.3f}"
}

print("\nKey Statistics:")
for key, value in summary_stats.items():
    print(f"  • {key}: {value}")

print("\n" + "="*80)
print("✅ STEP 2 COMPLETE - Exploratory Data Analysis")
print("="*80)
print(f"\n📁 Output files saved in: {OUTPUT_PATH.absolute()}")
print("\n📝 NEXT STEP:")
print("   → STEP 3: Advanced Feature Engineering")
print("     (Time-domain, frequency-domain, time-frequency, information theory)")
print("\n" + "="*80)