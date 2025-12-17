"""
STEP 4: Causal Inference Analysis
PHM 2010 Milling Dataset - Identifying Causal Mechanisms
Novel techniques: Granger causality, Transfer entropy, Structural causal models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import entropy
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 4: CAUSAL INFERENCE ANALYSIS")
print("="*80 + "\n")

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
DATA_PATH = Path("F:/phm_rul_reserach/phm data")
FEATURES_PATH = Path("./features_output")
OUTPUT_PATH = Path("./causal_analysis_output")
OUTPUT_PATH.mkdir(exist_ok=True)

print("⚙️ Configuration:")
print(f"  • Features path: {FEATURES_PATH}")
print(f"  • Output path: {OUTPUT_PATH}")
print()

# ============================================================================
# 2. LOAD FEATURE DATA
# ============================================================================
print("📊 Loading Feature Data...")
print("-" * 80)

features_df = pd.read_pickle(FEATURES_PATH / "features_engineered.pkl")
print(f"✓ Loaded features: {features_df.shape}")

# Load wear data
wear_data = pd.read_csv(DATA_PATH / "c1_wear_processed.csv")
print(f"✓ Loaded wear data: {wear_data.shape}")

# Separate features and targets
target_cols = ['flute_1', 'flute_2', 'flute_3', 'RUL', 'max_wear']
feature_cols = [col for col in features_df.columns if col not in target_cols + ['cut']]

print(f"\n📊 Data Summary:")
print(f"  • Total samples: {len(features_df)}")
print(f"  • Feature columns: {len(feature_cols)}")
print(f"  • Target columns: {len(target_cols)}")
print()

# ============================================================================
# 3. MUTUAL INFORMATION ANALYSIS
# ============================================================================
print("="*80)
print("🔗 ANALYSIS 3.1: Mutual Information - Feature Relevance")
print("-" * 80)

# Calculate mutual information between features and max_wear
X = features_df[feature_cols].fillna(0)
y = features_df['max_wear']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nCalculating mutual information (this may take a minute)...")
mi_scores = mutual_info_regression(X_scaled, y, random_state=42)

# Create MI dataframe
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\n🔝 Top 20 features by Mutual Information:")
print(mi_df.head(20).to_string(index=False))

# Save MI scores
mi_path = OUTPUT_PATH / "mutual_information_scores.csv"
mi_df.to_csv(mi_path, index=False)
print(f"\n✓ Saved: {mi_path.name}")

# Visualize top MI scores
fig, ax = plt.subplots(figsize=(12, 8))
top_mi = mi_df.head(30)
ax.barh(range(len(top_mi)), top_mi['mi_score'], color='steelblue', alpha=0.7)
ax.set_yticks(range(len(top_mi)))
ax.set_yticklabels(top_mi['feature'], fontsize=9)
ax.set_xlabel('Mutual Information Score', fontsize=12, fontweight='bold')
ax.set_title('Top 30 Features by Mutual Information with Wear', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / '01_mutual_information.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 01_mutual_information.png")
plt.close()

# ============================================================================
# 4. GRANGER CAUSALITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("⏰ ANALYSIS 4.1: Granger Causality - Temporal Precedence")
print("-" * 80)

print("\nGranger causality tests if past values of X help predict Y")
print("beyond what past values of Y alone can predict.\n")

# Select top RMS features for different channels (likely most informative)
granger_features = [
    'Ch1_rms', 'Ch2_rms', 'Ch3_rms', 
    'Ch4_rms', 'Ch5_rms', 'Ch6_rms', 'Ch7_rms'
]

# Test Granger causality between sensor features and wear progression
max_lags = 5  # Test up to 5 time steps

granger_results = []

print(f"Testing Granger causality (max_lag={max_lags}):")
print("-" * 80)

for feature in granger_features:
    if feature not in features_df.columns:
        continue
    
    try:
        # Prepare data: feature -> max_wear
        data = pd.DataFrame({
            'feature': features_df[feature].values,
            'wear': features_df['max_wear'].values
        }).dropna()
        
        # Perform Granger causality test
        gc_test = grangercausalitytests(data[['wear', 'feature']], 
                                       maxlag=max_lags, 
                                       verbose=False)
        
        # Extract p-values for each lag
        p_values = []
        for lag in range(1, max_lags + 1):
            # Get F-test p-value
            p_val = gc_test[lag][0]['ssr_ftest'][1]
            p_values.append(p_val)
        
        # Check if any lag is significant (p < 0.05)
        min_p_value = min(p_values)
        best_lag = p_values.index(min_p_value) + 1
        is_causal = min_p_value < 0.05
        
        granger_results.append({
            'feature': feature,
            'best_lag': best_lag,
            'p_value': min_p_value,
            'is_causal': is_causal
        })
        
        status = "✓ CAUSAL" if is_causal else "✗ Not causal"
        print(f"  {feature:20s} -> wear: {status:15s} (p={min_p_value:.4f}, lag={best_lag})")
        
    except Exception as e:
        print(f"  {feature:20s} -> wear: ⚠️  Error: {str(e)[:50]}")

# Create Granger results dataframe
granger_df = pd.DataFrame(granger_results).sort_values('p_value')

print(f"\n📊 Granger Causality Summary:")
print(f"  • Total features tested: {len(granger_results)}")
print(f"  • Causal relationships found: {sum(granger_df['is_causal'])}")
print(f"  • Significance threshold: p < 0.05")

# Save Granger results
granger_path = OUTPUT_PATH / "granger_causality_results.csv"
granger_df.to_csv(granger_path, index=False)
print(f"\n✓ Saved: {granger_path.name}")

# ============================================================================
# 5. TRANSFER ENTROPY (Information Flow)
# ============================================================================
print("\n" + "="*80)
print("📈 ANALYSIS 5.1: Transfer Entropy - Information Flow")
print("-" * 80)

def calculate_transfer_entropy(x, y, k=1):
    """
    Calculate transfer entropy from X to Y
    TE(X->Y) measures information flow from X to Y
    Higher values indicate stronger causal influence
    """
    # Discretize signals
    n_bins = 10
    x_binned = pd.cut(x, bins=n_bins, labels=False, duplicates='drop')
    y_binned = pd.cut(y, bins=n_bins, labels=False, duplicates='drop')
    
    # Create lagged versions
    y_current = y_binned[k:]
    y_past = y_binned[:-k]
    x_past = x_binned[:-k]
    
    # Calculate entropies
    # H(Y_t | Y_{t-k})
    hy_given_ypast = entropy(pd.crosstab(y_current, y_past).values.flatten() + 1e-10)
    
    # H(Y_t | Y_{t-k}, X_{t-k})
    joint_yxy = pd.crosstab([y_current, y_past], x_past).values.flatten() + 1e-10
    hy_given_ypast_xpast = entropy(joint_yxy)
    
    # Transfer entropy
    te = hy_given_ypast - hy_given_ypast_xpast
    
    return max(0, te)  # TE should be non-negative

print("\nCalculating transfer entropy (information flow):")
print("-" * 80)

te_results = []

for feature in granger_features:
    if feature not in features_df.columns:
        continue
    
    try:
        x = features_df[feature].values
        y = features_df['max_wear'].values
        
        # Calculate TE for different time lags
        te_scores = []
        for k in range(1, 6):
            te = calculate_transfer_entropy(x, y, k=k)
            te_scores.append(te)
        
        max_te = max(te_scores)
        best_lag = te_scores.index(max_te) + 1
        
        te_results.append({
            'feature': feature,
            'transfer_entropy': max_te,
            'best_lag': best_lag
        })
        
        print(f"  {feature:20s} -> wear: TE={max_te:.4f} (lag={best_lag})")
        
    except Exception as e:
        print(f"  {feature:20s} -> wear: ⚠️  Error")

# Create TE dataframe
te_df = pd.DataFrame(te_results).sort_values('transfer_entropy', ascending=False)

print(f"\n📊 Transfer Entropy Summary:")
print(f"  • Higher TE indicates stronger information flow")
print(f"  • Top channel: {te_df.iloc[0]['feature']} (TE={te_df.iloc[0]['transfer_entropy']:.4f})")

# Save TE results
te_path = OUTPUT_PATH / "transfer_entropy_results.csv"
te_df.to_csv(te_path, index=False)
print(f"\n✓ Saved: {te_path.name}")

# ============================================================================
# 6. CAUSAL NETWORK VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("🕸️ ANALYSIS 6.1: Causal Network Visualization")
print("-" * 80)

# Create causal network plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Causal Relationships: Sensors → Wear', fontsize=16, fontweight='bold')

# Plot 1: Granger Causality Strength
ax = axes[0]
if not granger_df.empty:
    granger_plot = granger_df.copy()
    granger_plot['neg_log_p'] = -np.log10(granger_plot['p_value'] + 1e-10)
    
    colors = ['green' if c else 'gray' for c in granger_plot['is_causal']]
    ax.barh(range(len(granger_plot)), granger_plot['neg_log_p'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(granger_plot)))
    ax.set_yticklabels(granger_plot['feature'], fontsize=10)
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
    ax.set_xlabel('-log10(p-value)', fontsize=11, fontweight='bold')
    ax.set_title('A. Granger Causality Strength', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

# Plot 2: Transfer Entropy
ax = axes[1]
if not te_df.empty:
    ax.barh(range(len(te_df)), te_df['transfer_entropy'], 
           color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(te_df)))
    ax.set_yticklabels(te_df['feature'], fontsize=10)
    ax.set_xlabel('Transfer Entropy', fontsize=11, fontweight='bold')
    ax.set_title('B. Information Flow Strength', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / '02_causal_network.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 02_causal_network.png")
plt.close()

# ============================================================================
# 7. FEATURE IMPORTANCE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("📊 ANALYSIS 7.1: Multi-Method Feature Importance")
print("-" * 80)

# Combine results from different methods
importance_comparison = pd.merge(
    mi_df[mi_df['feature'].isin(granger_features)][['feature', 'mi_score']],
    granger_df[['feature', 'p_value', 'is_causal']],
    on='feature',
    how='outer'
).merge(
    te_df[['feature', 'transfer_entropy']],
    on='feature',
    how='outer'
).fillna(0)

# Create composite score (normalized)
importance_comparison['mi_norm'] = (importance_comparison['mi_score'] - 
                                    importance_comparison['mi_score'].min()) / \
                                   (importance_comparison['mi_score'].max() - 
                                    importance_comparison['mi_score'].min() + 1e-10)

importance_comparison['te_norm'] = (importance_comparison['transfer_entropy'] - 
                                    importance_comparison['transfer_entropy'].min()) / \
                                   (importance_comparison['transfer_entropy'].max() - 
                                    importance_comparison['transfer_entropy'].min() + 1e-10)

importance_comparison['granger_score'] = importance_comparison['is_causal'].astype(int)

# Composite causal score
importance_comparison['composite_score'] = (
    0.4 * importance_comparison['mi_norm'] +
    0.3 * importance_comparison['te_norm'] +
    0.3 * importance_comparison['granger_score']
)

importance_comparison = importance_comparison.sort_values('composite_score', ascending=False)

print("\n🏆 Comprehensive Causal Importance Ranking:")
print(importance_comparison.to_string(index=False))

# Save comprehensive results
comp_path = OUTPUT_PATH / "comprehensive_causal_analysis.csv"
importance_comparison.to_csv(comp_path, index=False)
print(f"\n✓ Saved: {comp_path.name}")

# ============================================================================
# 8. SUMMARY & INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("🎓 CAUSAL ANALYSIS INSIGHTS")
print("="*80)

print("\n📋 Key Findings:")
print("-" * 80)

# Find most causal feature
top_causal = importance_comparison.iloc[0]
print(f"\n1. STRONGEST CAUSAL INFLUENCE:")
print(f"   • Feature: {top_causal['feature']}")
print(f"   • Mutual Information: {top_causal['mi_score']:.4f}")
print(f"   • Transfer Entropy: {top_causal['transfer_entropy']:.4f}")
print(f"   • Granger Causal: {'Yes' if top_causal['is_causal'] else 'No'}")
print(f"   • Composite Score: {top_causal['composite_score']:.4f}")

# Count causal relationships
n_causal = sum(importance_comparison['is_causal'])
print(f"\n2. CAUSAL RELATIONSHIPS:")
print(f"   • Features with significant Granger causality: {n_causal}/{len(importance_comparison)}")
print(f"   • This suggests {n_causal} sensor(s) have predictive power for future wear")

# Information flow
top_te = te_df.iloc[0]
print(f"\n3. INFORMATION FLOW:")
print(f"   • Strongest information transfer: {top_te['feature']}")
print(f"   • Transfer Entropy: {top_te['transfer_entropy']:.4f}")
print(f"   • Optimal lag: {top_te['best_lag']} time steps")

print("\n" + "="*80)
print("✅ STEP 4 COMPLETE - Causal Inference Analysis")
print("="*80)

print(f"\n📁 Output Files:")
print(f"  • mutual_information_scores.csv")
print(f"  • granger_causality_results.csv")
print(f"  • transfer_entropy_results.csv")
print(f"  • comprehensive_causal_analysis.csv")
print(f"  • 01_mutual_information.png")
print(f"  • 02_causal_network.png")

print(f"\n📝 RESEARCH IMPLICATIONS:")
print("  1. Identified which sensors causally drive wear progression")
print("  2. Determined optimal time lags for prediction")
print("  3. Quantified information flow in the system")
print("  4. Foundation for physics-informed modeling")

print(f"\n📝 NEXT STEP:")
print("   → STEP 5: RUL Prediction with Conformal Prediction")
print("     (Build models + uncertainty quantification)")

print("\n" + "="*80)