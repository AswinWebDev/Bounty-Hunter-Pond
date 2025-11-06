#!/usr/bin/env python
"""
Build Multi-Modal Sybil Detection Dataset

Combines:
1. Lightweight on-chain features (minimal API usage)
2. Social graph features (GitHub, Twitter, LinkedIn)
3. Behavioral features (if Pond bounty data available)

Output: Complete feature matrix ready for training
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_addresses(path: str) -> pd.DataFrame:
    """Load addresses from parquet or CSV"""
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_multimodal_features(
    addresses_path: str,
    onchain_features_path: str = None,
    social_features_path: str = None,
    output_path: str = None
) -> pd.DataFrame:
    """Combine all feature types into single dataframe"""
    
    print("=" * 60)
    print("Building Multi-Modal Feature Matrix")
    print("=" * 60)
    
    # Load addresses
    addresses_df = load_addresses(addresses_path)
    if 'ADDRESS' in addresses_df.columns:
        addresses_df['address'] = addresses_df['ADDRESS'].str.lower()
    elif 'address' not in addresses_df.columns:
        addresses_df['address'] = addresses_df.iloc[:, 0].str.lower()
    
    print(f"\n✓ Loaded {len(addresses_df)} addresses from {addresses_path}")
    
    # Start with addresses
    feature_df = addresses_df[['address']].copy()
    
    # Merge on-chain features
    if onchain_features_path and Path(onchain_features_path).exists():
        onchain_df = pd.read_csv(onchain_features_path)
        feature_df = feature_df.merge(onchain_df, on='address', how='left')
        print(f"✓ Merged {len(onchain_df.columns)} on-chain features")
    else:
        print(f"⚠️  No on-chain features found at {onchain_features_path}")
        # Add placeholder on-chain features
        feature_df['has_onchain_data'] = False
    
    # Merge social features
    if social_features_path and Path(social_features_path).exists():
        social_df = pd.read_csv(social_features_path)
        feature_df = feature_df.merge(social_df, on='address', how='left')
        print(f"✓ Merged {len(social_df.columns)} social features")
    else:
        print(f"⚠️  No social features found at {social_features_path}")
        # Add placeholder social features
        feature_df['has_social_data'] = False
    
    # Fill NaN values
    feature_df = feature_df.fillna(0)
    
    # Calculate composite scores
    print("\n📊 Calculating composite Sybil scores...")
    
    # On-chain risk score
    onchain_signals = []
    if 'new_account' in feature_df.columns:
        onchain_signals.append('new_account')
    if 'burst_activity' in feature_df.columns:
        onchain_signals.append('burst_activity')
    if 'dust_transactions' in feature_df.columns:
        onchain_signals.append('dust_transactions')
    if 'low_method_diversity' in feature_df.columns:
        onchain_signals.append('low_method_diversity')
    
    if onchain_signals:
        feature_df['onchain_risk_score'] = feature_df[onchain_signals].sum(axis=1) / len(onchain_signals)
        print(f"  On-chain risk: {feature_df['onchain_risk_score'].mean():.3f} avg")
    
    # Social risk score
    social_signals = []
    if 'github_suspicious' in feature_df.columns:
        social_signals.append('github_suspicious')
    if 'twitter_suspicious' in feature_df.columns:
        social_signals.append('twitter_suspicious')
    if 'single_platform_only' in feature_df.columns:
        social_signals.append('single_platform_only')
    
    if social_signals:
        feature_df['social_risk_score'] = feature_df[social_signals].sum(axis=1) / len(social_signals)
        print(f"  Social risk: {feature_df['social_risk_score'].mean():.3f} avg")
    
    # Combined risk score
    risk_components = []
    if 'onchain_risk_score' in feature_df.columns:
        risk_components.append('onchain_risk_score')
    if 'social_risk_score' in feature_df.columns:
        risk_components.append('social_risk_score')
    
    if risk_components:
        feature_df['combined_risk_score'] = feature_df[risk_components].mean(axis=1)
        print(f"  Combined risk: {feature_df['combined_risk_score'].mean():.3f} avg")
    
    # Save
    if output_path:
        out_path = Path(output_path)
    else:
        out_path = ROOT / 'data' / 'processed' / 'multimodal_features.csv'
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out_path, index=False)
    
    print(f"\n✅ Saved {len(feature_df)} × {len(feature_df.columns)} feature matrix to {out_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Feature Matrix Summary")
    print("=" * 60)
    print(f"Total addresses: {len(feature_df):,}")
    print(f"Total features: {len(feature_df.columns)}")
    print(f"\nFeature categories:")
    
    # Count feature types
    onchain_cols = [c for c in feature_df.columns if any(x in c for x in ['tx_', 'gas_', 'value_', 'dex_', 'method_'])]
    social_cols = [c for c in feature_df.columns if any(x in c for x in ['github_', 'twitter_', 'linkedin_'])]
    risk_cols = [c for c in feature_df.columns if 'risk' in c or 'suspicious' in c]
    
    print(f"  On-chain features: {len(onchain_cols)}")
    print(f"  Social features: {len(social_cols)}")
    print(f"  Risk scores: {len(risk_cols)}")
    
    if 'combined_risk_score' in feature_df.columns:
        print(f"\n🚨 High-risk addresses (risk > 0.7): {(feature_df['combined_risk_score'] > 0.7).sum():,}")
        print(f"🟢 Low-risk addresses (risk < 0.3): {(feature_df['combined_risk_score'] < 0.3).sum():,}")
    
    return feature_df


def main():
    parser = argparse.ArgumentParser(description="Build multi-modal Sybil detection dataset")
    parser.add_argument('--addresses', required=True, help='Path to addresses file')
    parser.add_argument('--onchain', help='Path to on-chain features CSV')
    parser.add_argument('--social', help='Path to social features CSV')
    parser.add_argument('--output', help='Output path for combined features')
    
    args = parser.parse_args()
    
    # Build feature matrix
    feature_df = build_multimodal_features(
        addresses_path=args.addresses,
        onchain_features_path=args.onchain or 'data/processed/light_onchain_base.csv',
        social_features_path=args.social or 'data/processed/social_features.csv',
        output_path=args.output
    )
    
    print("\n" + "=" * 60)
    print("✅ Multi-modal dataset build complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train model: python scripts/train_multimodal.py")
    print("2. Or use your hybrid_ensemble: python scripts/hybrid_ensemble.py")


if __name__ == '__main__':
    main()
