#!/usr/bin/env python
"""
Lightweight on-chain data collector - Minimal API usage

Strategy:
- Only fetch 10 most recent transactions per address
- Extract high-signal features from minimal data
- Cache everything aggressively
- 20k addresses × 10 tx = 200k API calls (fits in free tier over 2 days)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.explorer_client import EtherscanLikeClient


CACHE_DIR = ROOT / "cache" / "light_onchain"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_address_cache_path(address: str, chain: str) -> Path:
    """Get cache file path for an address"""
    return CACHE_DIR / chain / f"{address.lower()}.json"


def load_from_cache(address: str, chain: str) -> Dict:
    """Load cached data for an address"""
    cache_path = get_address_cache_path(address, chain)
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None


def save_to_cache(address: str, chain: str, data: Dict):
    """Save data to cache"""
    cache_path = get_address_cache_path(address, chain)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)


def extract_light_features(address: str, txs: List[Dict], chain: str) -> Dict:
    """Extract high-signal features from minimal transaction data"""
    
    if not txs:
        return {
            'address': address.lower(),
            'chain': chain,
            'tx_count': 0,
            'has_activity': False,
        }
    
    # Basic counts
    tx_count = len(txs)
    out_count = sum(1 for tx in txs if tx.get('from', '').lower() == address.lower())
    in_count = tx_count - out_count
    
    # Value analysis
    values = []
    for tx in txs:
        try:
            val = tx.get('value', 0)
            if val and isinstance(val, (int, float, str)):
                v = float(val)
                if 0 <= v < 1e20:  # Filter extremes
                    values.append(v)
        except (ValueError, TypeError):
            pass
    
    # Gas analysis
    gas_prices = []
    for tx in txs:
        try:
            gp = tx.get('gasPrice', 0)
            if gp and isinstance(gp, (int, float, str)):
                g = float(gp)
                if 0 <= g < 1e20:  # Filter extremes
                    gas_prices.append(g)
        except (ValueError, TypeError):
            pass
    
    # Timestamps
    timestamps = [int(tx.get('timeStamp', 0)) for tx in txs if 'timeStamp' in tx]
    timestamps = sorted([t for t in timestamps if t > 0])
    
    # Calculate intervals
    intervals = []
    if len(timestamps) > 1:
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    
    # Contract interactions
    to_addresses = [tx.get('to', '').lower() for tx in txs if tx.get('to')]
    unique_to = set(to_addresses)
    
    from_addresses = [tx.get('from', '').lower() for tx in txs if tx.get('from')]
    unique_from = set(from_addresses)
    
    # Known contracts (DEX, bridges, etc.)
    known_contracts = {
        # Base DEXs
        '0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24': 'BaseSwap',
        '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA': 'Uniswap_Base',
        '0x4200000000000000000000000000000000000006': 'WETH_Base',
        
        # Bridges
        '0x49048044d57e1c92a77f79988d21fa8faf74e97e': 'Base_Bridge',
    }
    
    dex_interactions = sum(1 for addr in to_addresses if addr in known_contracts)
    
    # Method signatures (first 10 chars of input data)
    methods = [tx.get('input', '')[:10] for tx in txs if tx.get('input')]
    unique_methods = set(methods)
    method_diversity = len(unique_methods) / max(len(methods), 1)
    
    # Build feature dict
    features = {
        'address': address.lower(),
        'chain': chain,
        'has_activity': tx_count > 0,
        
        # Transaction counts
        'tx_count': tx_count,
        'tx_out_count': out_count,
        'tx_in_count': in_count,
        'in_out_ratio': in_count / max(out_count, 1),
        
        # Network diversity
        'unique_to_addresses': len(unique_to),
        'unique_from_addresses': len(unique_from),
        'total_unique_counterparties': len(unique_to | unique_from),
        
        # Value statistics
        'value_mean': sum(values) / len(values) if values else 0,
        'value_sum': sum(values) if values else 0,
        'value_max': max(values) if values else 0,
        
        # Gas statistics
        'gas_price_mean': sum(gas_prices) / len(gas_prices) if gas_prices else 0,
        'gas_price_consistent': (max(gas_prices) - min(gas_prices)) / max(gas_prices, 1) < 0.1 if len(gas_prices) > 1 else False,
        
        # Temporal patterns
        'timestamp_first': timestamps[0] if timestamps else 0,
        'timestamp_last': timestamps[-1] if timestamps else 0,
        'account_age_seconds': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
        'interval_mean': sum(intervals) / len(intervals) if intervals else 0,
        'interval_std': pd.Series(intervals).std() if len(intervals) > 1 else 0,
        'regular_intervals': pd.Series(intervals).std() / max(pd.Series(intervals).mean(), 1) < 0.3 if len(intervals) > 1 else False,
        
        # Contract interactions
        'dex_interactions': dex_interactions,
        'has_dex_activity': dex_interactions > 0,
        
        # Method diversity
        'unique_methods': len(unique_methods),
        'method_diversity': method_diversity,
        'low_method_diversity': method_diversity < 0.3,
        
        # Sybil signals
        'new_account': (timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0) < 86400 * 30,  # < 30 days
        'burst_activity': len(timestamps) > 5 and (timestamps[-1] - timestamps[0]) < 3600,  # Many tx in 1 hour
        'dust_transactions': sum(1 for v in values if v < 0.001) / max(len(values), 1) > 0.8,  # Mostly dust
    }
    
    return features


def collect_light_data(addresses: List[str], chain: str, limit: int = 10, skip_cached: bool = True) -> pd.DataFrame:
    """Collect minimal on-chain data for addresses"""
    
    client = EtherscanLikeClient.for_chain(chain)
    results = []
    
    print(f"Collecting light on-chain data for {len(addresses)} addresses on {chain}")
    print(f"Fetching {limit} recent transactions per address")
    print(f"Total API calls: {len(addresses)} (caching enabled)")
    
    for addr in tqdm(addresses, desc=f"Fetching {chain} data"):
        addr = addr.lower().strip()
        
        # Check cache first
        if skip_cached:
            cached = load_from_cache(addr, chain)
            if cached:
                results.append(cached)
                continue
        
        try:
            # Fetch only recent transactions (minimal API usage)
            txs = client.account_txlist(addr, page=1, offset=limit, sort='desc')
            
            if not isinstance(txs, list):
                txs = []
            
            # Extract features
            features = extract_light_features(addr, txs, chain)
            
            # Cache result
            save_to_cache(addr, chain, features)
            results.append(features)
            
        except Exception as e:
            print(f"Error fetching {addr}: {e}")
            # Save empty result to avoid re-fetching
            empty_features = extract_light_features(addr, [], chain)
            save_to_cache(addr, chain, empty_features)
            results.append(empty_features)
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Collect lightweight on-chain features")
    parser.add_argument('--addresses', required=True, help='Path to addresses file (CSV/parquet)')
    parser.add_argument('--chain', default='base', choices=['ethereum', 'base'], help='Blockchain')
    parser.add_argument('--limit', type=int, default=10, help='Transactions per address')
    parser.add_argument('--output', help='Output path (default: data/processed/light_onchain_{chain}.csv)')
    parser.add_argument('--no-cache', action='store_true', help='Ignore cache and re-fetch')
    
    args = parser.parse_args()
    
    # Load addresses
    if args.addresses.endswith('.parquet'):
        df = pd.read_parquet(args.addresses)
    else:
        df = pd.read_csv(args.addresses)
    
    if 'ADDRESS' in df.columns:
        addresses = df['ADDRESS'].tolist()
    elif 'address' in df.columns:
        addresses = df['address'].tolist()
    else:
        addresses = df.iloc[:, 0].tolist()
    
    print(f"Loaded {len(addresses)} addresses from {args.addresses}")
    
    # Collect data
    features_df = collect_light_data(
        addresses, 
        args.chain, 
        limit=args.limit,
        skip_cached=not args.no_cache
    )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ROOT / 'data' / 'processed' / f'light_onchain_{args.chain}.csv'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved {len(features_df)} feature vectors to {output_path}")
    print(f"\n📊 Summary:")
    print(f"  Active addresses: {features_df['has_activity'].sum():,}")
    print(f"  Avg transactions: {features_df['tx_count'].mean():.1f}")
    
    # Only print if columns exist (addresses may have no activity)
    if 'has_dex_activity' in features_df.columns:
        print(f"  DEX users: {features_df['has_dex_activity'].sum():,}")
    if 'new_account' in features_df.columns:
        print(f"  Suspicious signals:")
        print(f"    - New accounts: {features_df.get('new_account', pd.Series([False])).sum():,}")
        print(f"    - Burst activity: {features_df.get('burst_activity', pd.Series([False])).sum():,}")
        print(f"    - Dust transactions: {features_df.get('dust_transactions', pd.Series([False])).sum():,}")
        print(f"    - Low method diversity: {features_df.get('low_method_diversity', pd.Series([False])).sum():,}")


if __name__ == '__main__':
    main()
