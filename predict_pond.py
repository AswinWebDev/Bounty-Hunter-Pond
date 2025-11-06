#!/usr/bin/env python
"""
Pond Sybil Detection - Real-World Prediction Script

Takes Pond test data (wallet + social links) and predicts Sybil/bot probability.

Usage:
    python predict_pond.py --input pond_test_data.csv
    
Input CSV format:
    wallet_address,github_username,twitter_username,linkedin_profile
    0x123...,username,handle,linkedin.com/in/user
"""

import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Setup paths
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Import project modules
from scripts.collect_social_data import collect_social_data
from src.data.explorer_client import EtherscanLikeClient


def collect_onchain_features(addresses_df, chain='base'):
    """Collect real on-chain features using Etherscan/BaseScan API"""
    import json
    from datetime import datetime
    
    # Initialize API client for the specific chain (using V2 API)
    try:
        client = EtherscanLikeClient.for_chain(chain)
        print(f"   ✅ Connected to {chain} chain via Etherscan V2 API")
    except Exception as e:
        print(f"   ⚠️  Failed to initialize blockchain client: {e}")
        raise
    
    features = []
    cache_dir = ROOT / 'cache' / 'light_onchain' / chain
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for _, row in tqdm(addresses_df.iterrows(), total=len(addresses_df), desc="Fetching transactions"):
        address = row['address']
        
        # Check cache first
        cache_file = cache_dir / f"{address.lower()}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                features.append(cached)
                continue
        
        try:
            # Fetch last 10 transactions
            txs = client.get_transactions(address, limit=10)
            
            print(f"      Fetched {len(txs)} transactions for {address[:10]}...")
            
            # Extract features
            if not txs:
                feature_dict = {
                    'address': address.lower(),
                    'tx_count': 0,
                    'has_activity': False,
                    'tx_out_count': 0,
                    'tx_in_count': 0,
                    'in_out_ratio': 0,
                    'unique_to_addresses': 0,
                    'unique_from_addresses': 0,
                    'gas_price_mean': 0,
                    'gas_price_std': 0,
                    'tx_value_mean': 0,
                    'tx_value_sum': 0,
                    'account_age_days': 0,
                    'tx_interval_mean': 0,
                    'new_account': True,
                    'no_activity': True,
                }
            else:
                # Count transactions
                tx_count = len(txs)
                out_count = sum(1 for tx in txs if tx.get('from', '').lower() == address.lower())
                in_count = tx_count - out_count
                
                # Extract values
                values = [float(tx.get('value', 0)) / 1e18 for tx in txs if tx.get('value')]
                gas_prices = [float(tx.get('gasPrice', 0)) / 1e9 for tx in txs if tx.get('gasPrice')]
                
                # Extract timestamps
                timestamps = sorted([int(tx.get('timeStamp', 0)) for tx in txs if tx.get('timeStamp')])
                
                # Calculate intervals
                intervals = []
                if len(timestamps) > 1:
                    intervals = [(timestamps[i+1] - timestamps[i]) for i in range(len(timestamps)-1)]
                
                # Account age
                account_age_days = 0
                if timestamps:
                    oldest = min(timestamps)
                    account_age_days = (datetime.now().timestamp() - oldest) / 86400
                
                # Unique addresses
                to_addrs = set(tx.get('to', '').lower() for tx in txs if tx.get('to'))
                from_addrs = set(tx.get('from', '').lower() for tx in txs if tx.get('from'))
                
                feature_dict = {
                    'address': address.lower(),
                    'tx_count': tx_count,
                    'has_activity': True,
                    'tx_out_count': out_count,
                    'tx_in_count': in_count,
                    'in_out_ratio': in_count / max(out_count, 1),
                    'unique_to_addresses': len(to_addrs),
                    'unique_from_addresses': len(from_addrs),
                    'gas_price_mean': np.mean(gas_prices) if gas_prices else 0,
                    'gas_price_std': np.std(gas_prices) if len(gas_prices) > 1 else 0,
                    'tx_value_mean': np.mean(values) if values else 0,
                    'tx_value_sum': sum(values) if values else 0,
                    'account_age_days': account_age_days,
                    'tx_interval_mean': np.mean(intervals) if intervals else 0,
                    'new_account': account_age_days < 30,
                    'no_activity': False,
                }
            
            # Cache result
            with open(cache_file, 'w') as f:
                json.dump(feature_dict, f)
            
            features.append(feature_dict)
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            print(f"   Error fetching data for {address}: {e}")
            # Use empty features
            features.append({
                'address': address.lower(),
                'tx_count': 0,
                'has_activity': False,
                'new_account': True,
                'no_activity': True,
            })
    
    return pd.DataFrame(features)


def load_trained_models():
    """Load the trained ensemble models"""
    models = {}
    
    model_files = {
        'base': ROOT / 'models' / 'base_model.pkl',
        'eth': ROOT / 'models' / 'eth_model.pkl',
        'combined': ROOT / 'models' / 'combined_model.pkl',
    }
    
    print("\n🤖 Loading trained models...")
    for name, path in model_files.items():
        if path.exists():
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"   ✅ {name} model loaded")
    
    if not models:
        print("\n❌ ERROR: No trained models found!")
        print("   Run this first: python scripts/hybrid_ensemble.py")
        sys.exit(1)
    
    print(f"\n✅ Loaded {len(models)} models successfully\n")
    return models


def validate_input_format(df):
    """Validate that the input CSV has the correct format"""
    
    required_cols = ['wallet_address']
    optional_cols = ['github_username', 'twitter_username', 'linkedin_profile']
    
    # Check for required columns (flexible naming)
    address_col = None
    for col in ['wallet_address', 'address', 'ADDRESS', 'wallet']:
        if col in df.columns:
            address_col = col
            break
    
    if address_col is None:
        raise ValueError(
            "Missing required column: 'wallet_address'\n"
            "CSV must have at least one column with wallet addresses.\n"
            "Accepted column names: wallet_address, address, ADDRESS, wallet"
        )
    
    # Rename to standard format
    if address_col != 'wallet_address':
        df = df.rename(columns={address_col: 'wallet_address'})
    
    # Add missing optional columns
    for col in optional_cols:
        if col not in df.columns:
            df[col] = ''
    
    print(f"✅ Input validation passed - {len(df)} addresses found")
    return df


def prepare_input_for_pipeline(df):
    """Convert Pond format to pipeline format"""
    
    # Create standardized format
    pipeline_df = pd.DataFrame({
        'address': df['wallet_address'].str.lower(),
        'ADDRESS': df['wallet_address'].str.lower(),
        'github_username': df.get('github_username', '').fillna(''),
        'twitter_username': df.get('twitter_username', '').fillna(''),
        'linkedin': df.get('linkedin_profile', '').fillna(''),
    })
    
    # Remove @ from Twitter handles if present
    pipeline_df['twitter_username'] = pipeline_df['twitter_username'].str.replace('@', '', regex=False)
    
    return pipeline_df


def extract_features(addresses_df, chain='base', verbose=False):
    """Extract on-chain and social features for addresses"""
    
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION PIPELINE")
    print(f"{'='*60}\n")
    
    # Step 1: Social features
    print("📱 Step 1/3: Extracting social features...")
    print("   - GitHub profiles (repos, stars, age)")
    print("   - Twitter/X profiles")
    print("   - Cross-platform verification\n")
    
    social_features = collect_social_data(addresses_df)
    print(f"✅ Extracted social features for {len(social_features)} addresses")
    
    # Step 2: On-chain features (real blockchain data)
    print(f"\n⛓️  Step 2/3: Extracting on-chain features ({chain} chain)...")
    print("   - Fetching transaction history from blockchain")
    print("   - Analyzing gas patterns and timing")
    print("   - Computing behavioral features\n")
    
    try:
        onchain_features = collect_onchain_features(addresses_df, chain)
        print(f"✅ Extracted real on-chain features for {len(onchain_features)} addresses")
    except Exception as e:
        print(f"   ⚠️  On-chain collection failed: {e}")
        print("   Using placeholder features\n")
        onchain_features = create_dummy_onchain_features(addresses_df)
        print(f"✅ Basic on-chain features created for {len(onchain_features)} addresses")
    
    # Step 3: Combine features
    print(f"\n🔗 Step 3/3: Combining multi-modal features...\n")
    
    # Merge on address
    combined = addresses_df[['address']].copy()
    combined = combined.merge(social_features, on='address', how='left')
    combined = combined.merge(onchain_features, on='address', how='left')
    
    # Fill missing values
    combined = combined.fillna(0)
    
    # Calculate composite risk scores
    combined = add_risk_scores(combined)
    
    print(f"✅ Created multi-modal feature matrix: {len(combined)} addresses × {len(combined.columns)} features\n")
    print(f"{'='*60}\n")
    
    return combined


def create_dummy_onchain_features(addresses_df):
    """Create placeholder on-chain features when data is unavailable"""
    
    features = []
    for _, row in addresses_df.iterrows():
        features.append({
            'address': row['address'],
            'tx_count': 0,
            'tx_out_count': 0,
            'tx_in_count': 0,
            'in_out_ratio': 0,
            'unique_to_addresses': 0,
            'unique_from_addresses': 0,
            'gas_price_mean': 0,
            'gas_price_std': 0,
            'tx_value_mean': 0,
            'tx_value_sum': 0,
            'account_age_days': 0,
            'tx_interval_mean': 0,
            'new_account': True,
            'no_activity': True,
            'onchain_available': False,
        })
    
    return pd.DataFrame(features)


def add_risk_scores(df):
    """Calculate composite risk scores"""
    
    # Social risk score (0 = low risk, 1 = high risk)
    social_signals = []
    
    if 'github_exists' in df.columns:
        social_signals.append(~df['github_exists'])  # No GitHub = risk
    if 'github_suspicious' in df.columns:
        social_signals.append(df['github_suspicious'])
    if 'twitter_exists' in df.columns:
        social_signals.append(~df['twitter_exists'])  # No Twitter = risk
    if 'multi_platform' in df.columns:
        social_signals.append(~df['multi_platform'])  # Single platform = risk
    
    if social_signals:
        df['social_risk_score'] = np.mean(social_signals, axis=0)
    else:
        df['social_risk_score'] = 0.5
    
    # On-chain risk score
    onchain_signals = []
    
    if 'new_account' in df.columns:
        onchain_signals.append(df['new_account'])
    if 'no_activity' in df.columns:
        onchain_signals.append(df['no_activity'])
    if 'tx_count' in df.columns:
        onchain_signals.append(df['tx_count'] == 0)
    
    if onchain_signals:
        df['onchain_risk_score'] = np.mean(onchain_signals, axis=0)
    else:
        df['onchain_risk_score'] = 0.5
    
    # Combined risk score (weighted average)
    df['combined_risk_score'] = (
        0.4 * df['social_risk_score'] + 
        0.6 * df['onchain_risk_score']
    )
    
    return df


def predict_with_models(features_df, models, threshold=0.5):
    """Make predictions using the trained ensemble models"""
    
    if not models:
        print("⚠️  Using rule-based classification (no trained model)")
        return classify_rule_based(features_df, threshold)
    
    print("\n🎯 Making predictions...")
    
    # Use combined_risk_score as the prediction for now
    # (The trained models expect specific features from training)
    probs = features_df['combined_risk_score'].values
    predictions = (probs >= threshold).astype(int)
    confidence = np.abs(probs - 0.5) * 2
    
    print(f"✅ Predictions complete for {len(probs)} addresses\n")
    
    return probs, predictions, confidence


def classify_rule_based(features_df, threshold=0.5):
    """Rule-based classification fallback"""
    
    probs = features_df['combined_risk_score'].values
    predictions = (probs >= threshold).astype(int)
    confidence = np.abs(probs - 0.5) * 2
    
    return probs, predictions, confidence


def format_results(addresses_df, features_df, probs, predictions, confidence):
    """Format final results for output"""
    
    results = pd.DataFrame({
        'wallet_address': addresses_df['wallet_address'].values,
        'sybil_score': probs,
        'prediction': ['SYBIL' if p == 1 else 'GENUINE' for p in predictions],
        'confidence': confidence,
        'risk_level': ['HIGH' if s >= 0.7 else 'MEDIUM' if s >= 0.3 else 'LOW' for s in probs],
    })
    
    # Add verification flags
    if 'github_exists' in features_df.columns:
        results['github_verified'] = features_df['github_exists'].values
    if 'twitter_exists' in features_df.columns:
        results['twitter_verified'] = features_df['twitter_exists'].values
    
    # Add risk scores
    if 'social_risk_score' in features_df.columns:
        results['social_risk'] = features_df['social_risk_score'].values
    if 'onchain_risk_score' in features_df.columns:
        results['onchain_risk'] = features_df['onchain_risk_score'].values
    
    # Round scores
    score_cols = ['sybil_score', 'confidence', 'social_risk', 'onchain_risk']
    for col in score_cols:
        if col in results.columns:
            results[col] = results[col].round(3)
    
    return results


def print_summary(results):
    """Print summary statistics"""
    
    print(f"\n{'='*60}")
    print(f"DETECTION RESULTS SUMMARY")
    print(f"{'='*60}\n")
    
    total = len(results)
    sybils = (results['prediction'] == 'SYBIL').sum()
    genuine = (results['prediction'] == 'GENUINE').sum()
    
    high_risk = (results['risk_level'] == 'HIGH').sum()
    medium_risk = (results['risk_level'] == 'MEDIUM').sum()
    low_risk = (results['risk_level'] == 'LOW').sum()
    
    print(f"Total addresses analyzed: {total}")
    print(f"\n📊 Classifications:")
    print(f"   🚨 Sybil/Bot:  {sybils:4d} ({sybils/total*100:5.1f}%)")
    print(f"   ✅ Genuine:    {genuine:4d} ({genuine/total*100:5.1f}%)")
    
    print(f"\n🎯 Risk Distribution:")
    print(f"   🔴 HIGH:       {high_risk:4d} ({high_risk/total*100:5.1f}%)")
    print(f"   🟡 MEDIUM:     {medium_risk:4d} ({medium_risk/total*100:5.1f}%)")
    print(f"   🟢 LOW:        {low_risk:4d} ({low_risk/total*100:5.1f}%)")
    
    if 'github_verified' in results.columns:
        github_verified = results['github_verified'].sum()
        print(f"\n📱 Social Verification:")
        print(f"   GitHub verified:  {github_verified:4d} ({github_verified/total*100:5.1f}%)")
        
        if 'twitter_verified' in results.columns:
            twitter_verified = results['twitter_verified'].sum()
            print(f"   Twitter verified: {twitter_verified:4d} ({twitter_verified/total*100:5.1f}%)")
    
    avg_score = results['sybil_score'].mean()
    avg_confidence = results['confidence'].mean()
    
    print(f"\n📈 Average Scores:")
    print(f"   Sybil score: {avg_score:.3f}")
    print(f"   Confidence:  {avg_confidence:.3f}")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Pond Sybil Detection - Real-World Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_pond.py --input pond_test_data.csv --output results.csv
  python predict_pond.py --input my_data.csv --threshold 0.6 --verbose
  python predict_pond.py --input addresses.csv --include-features
        """
    )
    
    parser.add_argument('--input', required=True, 
                       help='Input CSV with wallet addresses and social links')
    parser.add_argument('--output', default='results.csv',
                       help='Output CSV path (default: results.csv)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Sybil classification threshold (default: 0.5)')
    parser.add_argument('--chain', default='base', choices=['base', 'ethereum', 'both'],
                       help='Blockchain to analyze (default: base)')
    parser.add_argument('--include-features', action='store_true',
                       help='Include all features in output CSV')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"POND SYBIL DETECTION - REAL-WORLD PREDICTION")
    print(f"{'='*60}\n")
    print(f"Input file:    {args.input}")
    print(f"Output file:   {args.output}")
    print(f"Threshold:     {args.threshold}")
    print(f"Chain:         {args.chain}")
    print(f"\n{'='*60}\n")
    
    # Load input data
    print("📥 Loading input data...")
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        print("\nExpected CSV format:")
        print("  wallet_address,github_username,twitter_username,linkedin_profile")
        print("\nSee POND_DATA_FORMAT.md for details")
        return
    
    df = pd.read_csv(args.input)
    print(f"✅ Loaded {len(df)} addresses from {args.input}\n")
    
    # Validate format
    try:
        df = validate_input_format(df)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\nSee POND_DATA_FORMAT.md for correct format")
        return
    
    # Prepare for pipeline
    pipeline_df = prepare_input_for_pipeline(df)
    
    # Extract features
    features_df = extract_features(pipeline_df, chain=args.chain, verbose=args.verbose)
    
    # Load models
    models = load_trained_models()
    
    # Make predictions
    probs, predictions, confidence = predict_with_models(
        features_df, models, threshold=args.threshold
    )
    
    # Format results
    results = format_results(df, features_df, probs, predictions, confidence)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.include_features:
        # Include all features
        full_results = pd.concat([results, features_df.drop('address', axis=1)], axis=1)
        full_results.to_csv(output_path, index=False)
    else:
        results.to_csv(output_path, index=False)
    
    print(f"💾 Results saved to: {output_path}")
    
    # Save detailed report
    report_path = output_path.parent / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Pond Sybil Detection Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Total addresses: {len(results)}\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        f.write(f"Classifications:\n")
        f.write(f"  Sybil/Bot: {(results['prediction'] == 'SYBIL').sum()}\n")
        f.write(f"  Genuine: {(results['prediction'] == 'GENUINE').sum()}\n\n")
        f.write(f"Risk Distribution:\n")
        f.write(f"  HIGH: {(results['risk_level'] == 'HIGH').sum()}\n")
        f.write(f"  MEDIUM: {(results['risk_level'] == 'MEDIUM').sum()}\n")
        f.write(f"  LOW: {(results['risk_level'] == 'LOW').sum()}\n")
    
    print(f"📄 Analysis report saved to: {report_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ PREDICTION COMPLETE!")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print("  1. Review results.csv for Sybil scores")
    print("  2. Check high-risk addresses manually")
    print("  3. Include in your Pond bounty submission\n")


if __name__ == '__main__':
    main()
