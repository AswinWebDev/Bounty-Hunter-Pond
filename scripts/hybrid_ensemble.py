#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid Ensemble Model for Sybil Detection Competition

This model combines:
1. Chain-specific approach from improved_ensemble.py (Ethereum and Base chains)
2. Advanced feature engineering from script.py
3. Optimized for AUC (Area Under ROC Curve) instead of accuracy
4. Outputs probability scores (0 to 1) as required by new competition

The goal is to achieve similar performance to the original script.py (0.9782)
while meeting the requirements of the new competition.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import pyarrow.parquet as pq
from tqdm import tqdm
import warnings
import pickle
import time
from datetime import datetime
from collections import defaultdict, Counter

# Machine learning libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

# Start timing
start_time = time.time()

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hybrid_ensemble.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('hybrid_ensemble')

# Constants
ETHEREUM_DIR = 'Datasets/ethereum'
BASE_DIR = 'Datasets/base'
OUTPUT_DIR = 'models'
CACHE_DIR = 'cache/hybrid'
SEED = 42
np.random.seed(SEED)

# Ensemble weights
ETH_WEIGHT = 0.55  # Ethereum model weight
BASE_WEIGHT = 0.25  # Base model weight
COMBINED_WEIGHT = 0.20  # Combined model weight

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Define cache file paths
cache_eth_features = f'{CACHE_DIR}/eth_features.pkl'
cache_base_features = f'{CACHE_DIR}/base_features.pkl'
cache_combined_features = f'{CACHE_DIR}/combined_features.pkl'
cache_eth_model = f'{CACHE_DIR}/eth_model.pkl'
cache_base_model = f'{CACHE_DIR}/base_model.pkl'
cache_combined_model = f'{CACHE_DIR}/combined_model.pkl'
cache_meta_model = f'{CACHE_DIR}/meta_model.pkl'

# Helper Functions
def load_addresses():
    """Load training and test addresses from both chains"""
    try:
        logger.info("Loading addresses...")
        
        # Load Ethereum and Base training addresses
        eth_train = pq.read_table(f'{ETHEREUM_DIR}/train_addresses.parquet').to_pandas()
        base_train = pq.read_table(f'{BASE_DIR}/train_addresses.parquet').to_pandas()
        
        # Load test addresses from chain-specific directories
        eth_test = pq.read_table(f'{ETHEREUM_DIR}/test_addresses.parquet').to_pandas()
        base_test = pq.read_table(f'{BASE_DIR}/test_addresses.parquet').to_pandas()
        logger.info(f"Loaded test addresses from chain directories")
        
        # Combine datasets with chain information
        eth_train['CHAIN'] = 'ethereum'
        base_train['CHAIN'] = 'base'
        eth_test['CHAIN'] = 'ethereum'
        base_test['CHAIN'] = 'base'
        
        # Combine datasets
        all_train = pd.concat([eth_train, base_train]).drop_duplicates(subset=['ADDRESS'])
        all_test = pd.concat([eth_test, base_test]).drop_duplicates(subset=['ADDRESS'])
        
        # Ensure consistent lowercase addresses
        all_train['ADDRESS'] = all_train['ADDRESS'].str.lower()
        all_test['ADDRESS'] = all_test['ADDRESS'].str.lower()
        
        # Make sure LABEL is integer for training data
        if 'LABEL' in all_train.columns:
            all_train['LABEL'] = all_train['LABEL'].astype(int)
        
        logger.info(f"Loaded {len(all_train)} training addresses and {len(all_test)} test addresses")
        if 'LABEL' in all_train.columns:
            logger.info(f"Label distribution: {all_train['LABEL'].value_counts().to_dict()}")
        
        # Create separate dataframes for each chain
        eth_addresses = pd.concat([
            eth_train[['ADDRESS']].assign(is_train=True), 
            eth_test[['ADDRESS']].assign(is_train=False)
        ])
        
        base_addresses = pd.concat([
            base_train[['ADDRESS']].assign(is_train=True), 
            base_test[['ADDRESS']].assign(is_train=False)
        ])
        
        return all_train, all_test, eth_train, eth_test, base_train, base_test, eth_addresses, base_addresses
    
    except Exception as e:
        logger.error(f"Error loading addresses: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None, None, None, None, None


def extract_basic_features(chain_dir, addresses, chain_name):
    """Extract basic features for addresses on a specific chain
    
    Parameters:
    -----------
    chain_dir : str
        Directory containing the chain data
    addresses : pd.DataFrame
        DataFrame containing addresses to extract features for
    chain_name : str
        Name of the chain (ethereum or base)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with basic features for each address
    """
    try:
        logger.info(f"Extracting basic features from {chain_name}...")
        
        # Create a lookup set for faster checks
        address_set = set(addresses['ADDRESS'].str.lower())
        
        # Initialize feature dictionary
        feature_dict = {addr.lower(): {
            'tx_count': 0,
            'tx_out_count': 0, 
            'tx_in_count': 0,
            'gas_price_sum': 0,
            'gas_used_sum': 0,
            'tx_value_sum': 0,
            'unique_to_addresses': set(),
            'unique_from_addresses': set(),
            'contract_interactions': set(),
            'tx_intervals': [],
            'last_tx_time': None,
            'gas_price_values': [],
            'gas_used_values': [],
            'tx_value_values': [],
            'chain': chain_name
        } for addr in addresses['ADDRESS']}
        
        # Read transactions parquet
        tx_path = os.path.join(chain_dir, 'transactions.parquet')
        
        # Calculate batch size based on file size
        file_size_mb = os.path.getsize(tx_path) / (1024 * 1024)
        batch_size = min(100000, max(10000, int(200000000 / file_size_mb)))
        
        logger.info(f"Processing {tx_path} in batches of {batch_size}")
        
        # Open parquet file
        pf = pq.ParquetFile(tx_path)
        
        # Process in batches with tqdm progress bar
        total_batches = pf.num_row_groups
        for batch_i, batch in enumerate(tqdm(pf.iter_batches(batch_size=batch_size), 
                                          total=total_batches, 
                                          desc=f"Processing {chain_name} transactions")):
            # Convert to pandas
            batch_df = batch.to_pandas()
            
            # Ensure addresses are lowercase
            batch_df['FROM_ADDRESS'] = batch_df['FROM_ADDRESS'].str.lower()
            batch_df['TO_ADDRESS'] = batch_df['TO_ADDRESS'].str.lower()
            
            # Sort by timestamp if available
            if 'BLOCK_TIMESTAMP' in batch_df.columns:
                batch_df = batch_df.sort_values('BLOCK_TIMESTAMP')
            
            # Process transactions
            for _, tx in batch_df.iterrows():
                from_addr = tx['FROM_ADDRESS']
                to_addr = tx['TO_ADDRESS']
                
                # Skip if neither address is in our set
                if from_addr not in address_set and to_addr not in address_set:
                    continue
                
                # Get timestamp if available
                timestamp = None
                if 'BLOCK_TIMESTAMP' in tx and pd.notna(tx['BLOCK_TIMESTAMP']):
                    timestamp = pd.to_datetime(tx['BLOCK_TIMESTAMP'])
                
                # Process outgoing transactions
                if from_addr in address_set:
                    feature_dict[from_addr]['tx_count'] += 1
                    feature_dict[from_addr]['tx_out_count'] += 1
                    
                    # Add unique to address
                    if pd.notna(to_addr):
                        feature_dict[from_addr]['unique_to_addresses'].add(to_addr)
                        
                        # Track contract interactions (address starting with 0x but not ENS)
                        if to_addr.startswith('0x') and len(to_addr) == 42:
                            feature_dict[from_addr]['contract_interactions'].add(to_addr)
                    
                    # Add value if available
                    if 'VALUE' in tx and pd.notna(tx['VALUE']):
                        value = float(tx['VALUE'])
                        if 0 <= value < 1e20:  # Filter extreme values
                            feature_dict[from_addr]['tx_value_sum'] += value
                            feature_dict[from_addr]['tx_value_values'].append(value)
                    
                    # Add gas data if available
                    if 'GAS_PRICE' in tx and pd.notna(tx['GAS_PRICE']):
                        gas_price = float(tx['GAS_PRICE'])
                        if 0 <= gas_price < 1e20:  # Filter extreme values
                            feature_dict[from_addr]['gas_price_sum'] += gas_price
                            feature_dict[from_addr]['gas_price_values'].append(gas_price)
                    
                    if 'GAS_USED' in tx and pd.notna(tx['GAS_USED']):
                        gas_used = float(tx['GAS_USED'])
                        if 0 <= gas_used < 1e20:  # Filter extreme values
                            feature_dict[from_addr]['gas_used_sum'] += gas_used
                            feature_dict[from_addr]['gas_used_values'].append(gas_used)
                    
                    # Calculate time between transactions
                    if timestamp is not None and feature_dict[from_addr]['last_tx_time'] is not None:
                        interval = (timestamp - feature_dict[from_addr]['last_tx_time']).total_seconds()
                        if 0 <= interval < 31536000:  # Filter intervals > 1 year
                            feature_dict[from_addr]['tx_intervals'].append(interval)
                    
                    # Update last transaction time
                    if timestamp is not None:
                        feature_dict[from_addr]['last_tx_time'] = timestamp
                
                # Process incoming transactions
                if to_addr in address_set:
                    feature_dict[to_addr]['tx_count'] += 1
                    feature_dict[to_addr]['tx_in_count'] += 1
                    
                    # Add unique from address
                    if pd.notna(from_addr):
                        feature_dict[to_addr]['unique_from_addresses'].add(from_addr)
                    
                    # Add value if available and for incoming transactions
                    if 'VALUE' in tx and pd.notna(tx['VALUE']):
                        value = float(tx['VALUE'])
                        if 0 <= value < 1e20:  # Filter extreme values
                            # We're only tracking value received, not adding to tx_value_sum
                            # But we do keep track in tx_value_values for statistical purposes
                            feature_dict[to_addr]['tx_value_values'].append(value)
        
        # Process token transfers if available
        try:
            token_transfers_path = os.path.join(chain_dir, 'token_transfers.parquet')
            if os.path.exists(token_transfers_path):
                logger.info(f"Processing token transfers for {chain_name}...")
                
                # Add token transfer features to feature dictionary
                for addr in feature_dict:
                    feature_dict[addr].update({
                        'token_tx_count': 0,
                        'token_tx_out_count': 0,
                        'token_tx_in_count': 0,
                        'unique_tokens_sent': set(),
                        'unique_tokens_received': set(),
                        'token_recipients': set(),
                        'token_senders': set(),
                        'token_values': [],
                    })
                
                # Calculate batch size based on file size
                file_size_mb = os.path.getsize(token_transfers_path) / (1024 * 1024)
                batch_size = min(100000, max(10000, int(200000000 / file_size_mb)))
                
                # Open parquet file
                pf = pq.ParquetFile(token_transfers_path)
                
                # Process in batches with tqdm progress bar
                total_batches = pf.num_row_groups
                for batch_i, batch in enumerate(tqdm(pf.iter_batches(batch_size=batch_size), 
                                                total=total_batches, 
                                                desc=f"Processing {chain_name} token transfers")):
                    # Convert to pandas
                    batch_df = batch.to_pandas()
                    
                    # Ensure addresses are lowercase
                    batch_df['FROM_ADDRESS'] = batch_df['FROM_ADDRESS'].str.lower()
                    batch_df['TO_ADDRESS'] = batch_df['TO_ADDRESS'].str.lower()
                    if 'CONTRACT_ADDRESS' in batch_df.columns:
                        batch_df['CONTRACT_ADDRESS'] = batch_df['CONTRACT_ADDRESS'].str.lower()
                    
                    # Sort by timestamp if available
                    if 'BLOCK_TIMESTAMP' in batch_df.columns:
                        batch_df = batch_df.sort_values('BLOCK_TIMESTAMP')
                    
                    # Process token transfers
                    for _, transfer in batch_df.iterrows():
                        from_addr = transfer['FROM_ADDRESS']
                        to_addr = transfer['TO_ADDRESS']
                        
                        # Skip if neither address is in our set
                        if from_addr not in address_set and to_addr not in address_set:
                            continue
                        
                        token_addr = transfer.get('CONTRACT_ADDRESS', 'unknown')
                        
                        # Get token value if available
                        token_value = None
                        if 'AMOUNT_PRECISE' in transfer and pd.notna(transfer['AMOUNT_PRECISE']):
                            token_value = float(transfer['AMOUNT_PRECISE'])
                        elif 'AMOUNT_USD' in transfer and pd.notna(transfer['AMOUNT_USD']):
                            token_value = float(transfer['AMOUNT_USD'])
                        
                        # Process outgoing token transfers
                        if from_addr in address_set:
                            feature_dict[from_addr]['token_tx_count'] += 1
                            feature_dict[from_addr]['token_tx_out_count'] += 1
                            
                            # Add unique token and recipient
                            if pd.notna(token_addr):
                                feature_dict[from_addr]['unique_tokens_sent'].add(token_addr)
                            
                            if pd.notna(to_addr):
                                feature_dict[from_addr]['token_recipients'].add(to_addr)
                            
                            # Add token value
                            if token_value is not None and 0 <= token_value < 1e20:
                                feature_dict[from_addr]['token_values'].append(token_value)
                        
                        # Process incoming token transfers
                        if to_addr in address_set:
                            feature_dict[to_addr]['token_tx_count'] += 1
                            feature_dict[to_addr]['token_tx_in_count'] += 1
                            
                            # Add unique token and sender
                            if pd.notna(token_addr):
                                feature_dict[to_addr]['unique_tokens_received'].add(token_addr)
                            
                            if pd.notna(from_addr):
                                feature_dict[to_addr]['token_senders'].add(from_addr)
                            
                            # Add token value
                            if token_value is not None and 0 <= token_value < 1e20:
                                feature_dict[to_addr]['token_values'].append(token_value)
            else:
                logger.warning(f"Token transfers file not found for {chain_name}: {token_transfers_path}")
        except Exception as e:
            logger.error(f"Error processing token transfers for {chain_name}: {str(e)}")
        
        # Process DEX swaps if available
        try:
            dex_swaps_path = os.path.join(chain_dir, 'dex_swaps.parquet')
            if os.path.exists(dex_swaps_path):
                logger.info(f"Processing DEX swaps for {chain_name}...")
                
                # Add DEX swap features to feature dictionary
                for addr in feature_dict:
                    feature_dict[addr].update({
                        'dex_swap_count': 0,
                        'tokens_swapped_in': set(),
                        'tokens_swapped_out': set(),
                        'dex_swap_values_in': [],
                        'dex_swap_values_out': [],
                    })
                
                # Calculate batch size based on file size
                file_size_mb = os.path.getsize(dex_swaps_path) / (1024 * 1024)
                batch_size = min(100000, max(10000, int(200000000 / file_size_mb)))
                
                # Open parquet file
                pf = pq.ParquetFile(dex_swaps_path)
                
                # Process in batches with tqdm progress bar
                total_batches = pf.num_row_groups
                for batch_i, batch in enumerate(tqdm(pf.iter_batches(batch_size=batch_size), 
                                                total=total_batches, 
                                                desc=f"Processing {chain_name} DEX swaps")):
                    # Convert to pandas
                    batch_df = batch.to_pandas()
                    
                    # Ensure addresses are lowercase
                    if 'ORIGIN_FROM_ADDRESS' in batch_df.columns:
                        batch_df['ORIGIN_FROM_ADDRESS'] = batch_df['ORIGIN_FROM_ADDRESS'].str.lower()
                    if 'TX_TO' in batch_df.columns:
                        batch_df['TX_TO'] = batch_df['TX_TO'].str.lower()
                    if 'TOKEN_IN' in batch_df.columns:
                        batch_df['TOKEN_IN'] = batch_df['TOKEN_IN'].str.lower()
                    if 'TOKEN_OUT' in batch_df.columns:
                        batch_df['TOKEN_OUT'] = batch_df['TOKEN_OUT'].str.lower()
                    
                    # Sort by timestamp if available
                    if 'BLOCK_TIMESTAMP' in batch_df.columns:
                        batch_df = batch_df.sort_values('BLOCK_TIMESTAMP')
                    
                    # Process DEX swaps
                    for _, swap in batch_df.iterrows():
                        # Get addresses involved
                        from_addr = swap.get('ORIGIN_FROM_ADDRESS', None)
                        to_addr = swap.get('TX_TO', None)
                        
                        # Skip if neither address is in our set
                        if (from_addr is None or from_addr not in address_set) and \
                           (to_addr is None or to_addr not in address_set):
                            continue
                        
                        # Get token addresses
                        token_in = swap.get('TOKEN_IN', 'unknown')
                        token_out = swap.get('TOKEN_OUT', 'unknown')
                        
                        # Get swap values
                        value_in = None
                        if 'AMOUNT_IN_USD' in swap and pd.notna(swap['AMOUNT_IN_USD']):
                            value_in = float(swap['AMOUNT_IN_USD'])
                        elif 'AMOUNT_IN' in swap and pd.notna(swap['AMOUNT_IN']):
                            value_in = float(swap['AMOUNT_IN'])
                        
                        value_out = None
                        if 'AMOUNT_OUT_USD' in swap and pd.notna(swap['AMOUNT_OUT_USD']):
                            value_out = float(swap['AMOUNT_OUT_USD'])
                        elif 'AMOUNT_OUT' in swap and pd.notna(swap['AMOUNT_OUT']):
                            value_out = float(swap['AMOUNT_OUT'])
                        
                        # Process swap for from_address
                        if from_addr in address_set:
                            feature_dict[from_addr]['dex_swap_count'] += 1
                            
                            # Add tokens
                            if pd.notna(token_in):
                                feature_dict[from_addr]['tokens_swapped_in'].add(token_in)
                            if pd.notna(token_out):
                                feature_dict[from_addr]['tokens_swapped_out'].add(token_out)
                            
                            # Add values
                            if value_in is not None and 0 <= value_in < 1e20:
                                feature_dict[from_addr]['dex_swap_values_in'].append(value_in)
                            if value_out is not None and 0 <= value_out < 1e20:
                                feature_dict[from_addr]['dex_swap_values_out'].append(value_out)
                        
                        # Process swap for to_address if different from from_address
                        if to_addr in address_set and to_addr != from_addr:
                            feature_dict[to_addr]['dex_swap_count'] += 1
                            
                            # Add tokens
                            if pd.notna(token_in):
                                feature_dict[to_addr]['tokens_swapped_in'].add(token_in)
                            if pd.notna(token_out):
                                feature_dict[to_addr]['tokens_swapped_out'].add(token_out)
                            
                            # Add values
                            if value_in is not None and 0 <= value_in < 1e20:
                                feature_dict[to_addr]['dex_swap_values_in'].append(value_in)
                            if value_out is not None and 0 <= value_out < 1e20:
                                feature_dict[to_addr]['dex_swap_values_out'].append(value_out)
            else:
                logger.warning(f"DEX swaps file not found for {chain_name}: {dex_swaps_path}")
        except Exception as e:
            logger.error(f"Error processing DEX swaps for {chain_name}: {str(e)}")
        
        # Convert features to dataframe format
        logger.info("Converting features to dataframe...")
        result_data = []
        
        for addr, features in feature_dict.items():
            # Basic transaction features
            row = {
                'ADDRESS': addr,
                'tx_count': features['tx_count'],
                'tx_out_count': features['tx_out_count'],
                'tx_in_count': features['tx_in_count'],
                'in_out_ratio': features['tx_in_count'] / max(1, features['tx_out_count']),
                'unique_to_addresses': len(features['unique_to_addresses']),
                'unique_from_addresses': len(features['unique_from_addresses']),
                'unique_contacts': len(features['contract_interactions']),
                'chain': features['chain'],
            }
            
            # Add gas features if available
            if len(features['gas_price_values']) > 0:
                row['gas_price_mean'] = np.mean(features['gas_price_values'])
                row['gas_price_std'] = np.std(features['gas_price_values']) if len(features['gas_price_values']) > 1 else 0
                row['gas_price_median'] = np.median(features['gas_price_values'])
            else:
                row['gas_price_mean'] = 0
                row['gas_price_std'] = 0
                row['gas_price_median'] = 0
            
            if len(features['gas_used_values']) > 0:
                row['gas_used_mean'] = np.mean(features['gas_used_values'])
                row['gas_used_std'] = np.std(features['gas_used_values']) if len(features['gas_used_values']) > 1 else 0
                row['gas_used_median'] = np.median(features['gas_used_values'])
            else:
                row['gas_used_mean'] = 0
                row['gas_used_std'] = 0
                row['gas_used_median'] = 0
            
            # Add value features
            if len(features['tx_value_values']) > 0:
                row['tx_value_mean'] = np.mean(features['tx_value_values'])
                row['tx_value_std'] = np.std(features['tx_value_values']) if len(features['tx_value_values']) > 1 else 0
                row['tx_value_median'] = np.median(features['tx_value_values'])
                row['tx_value_max'] = max(features['tx_value_values'])
                row['tx_value_sum'] = sum(features['tx_value_values'])
            else:
                row['tx_value_mean'] = 0
                row['tx_value_std'] = 0
                row['tx_value_median'] = 0
                row['tx_value_max'] = 0
                row['tx_value_sum'] = 0
            
            # Add temporal features
            if len(features['tx_intervals']) > 0:
                row['tx_interval_mean'] = np.mean(features['tx_intervals'])
                row['tx_interval_std'] = np.std(features['tx_intervals']) if len(features['tx_intervals']) > 1 else 0
                row['tx_interval_median'] = np.median(features['tx_intervals'])
                row['tx_interval_min'] = min(features['tx_intervals'])
            else:
                row['tx_interval_mean'] = 0
                row['tx_interval_std'] = 0
                row['tx_interval_median'] = 0
                row['tx_interval_min'] = 0
            
            # Add token transfer features if available
            if 'token_tx_count' in features:
                row['token_tx_count'] = features['token_tx_count']
                row['token_tx_out_count'] = features['token_tx_out_count']
                row['token_tx_in_count'] = features['token_tx_in_count']
                row['unique_tokens_sent'] = len(features['unique_tokens_sent'])
                row['unique_tokens_received'] = len(features['unique_tokens_received'])
                row['token_recipients'] = len(features['token_recipients'])
                row['token_senders'] = len(features['token_senders'])
                
                if len(features['token_values']) > 0:
                    row['token_value_mean'] = np.mean(features['token_values'])
                    row['token_value_std'] = np.std(features['token_values']) if len(features['token_values']) > 1 else 0
                    row['token_value_median'] = np.median(features['token_values'])
                    row['token_value_max'] = max(features['token_values'])
                    row['token_value_sum'] = sum(features['token_values'])
                else:
                    row['token_value_mean'] = 0
                    row['token_value_std'] = 0
                    row['token_value_median'] = 0
                    row['token_value_max'] = 0
                    row['token_value_sum'] = 0
            
            # Add DEX swap features if available
            if 'dex_swap_count' in features:
                row['dex_swap_count'] = features['dex_swap_count']
                row['tokens_swapped_in'] = len(features['tokens_swapped_in'])
                row['tokens_swapped_out'] = len(features['tokens_swapped_out'])
                
                if len(features['dex_swap_values_in']) > 0:
                    row['dex_swap_in_mean'] = np.mean(features['dex_swap_values_in'])
                    row['dex_swap_in_sum'] = sum(features['dex_swap_values_in'])
                else:
                    row['dex_swap_in_mean'] = 0
                    row['dex_swap_in_sum'] = 0
                
                if len(features['dex_swap_values_out']) > 0:
                    row['dex_swap_out_mean'] = np.mean(features['dex_swap_values_out'])
                    row['dex_swap_out_sum'] = sum(features['dex_swap_values_out'])
                else:
                    row['dex_swap_out_mean'] = 0
                    row['dex_swap_out_sum'] = 0
            
            # Add the row to the result data
            result_data.append(row)
        
        # Create dataframe from result data
        result_df = pd.DataFrame(result_data)
        
        # Add derived features
        if 'tx_count' in result_df.columns:
            # Activity patterns
            result_df['activity_ratio'] = result_df['tx_out_count'] / result_df['tx_count'].apply(lambda x: max(x, 1))
            
            # Network-based features
            result_df['avg_to_per_tx'] = result_df['unique_to_addresses'] / result_df['tx_out_count'].apply(lambda x: max(x, 1))
            result_df['avg_from_per_tx'] = result_df['unique_from_addresses'] / result_df['tx_in_count'].apply(lambda x: max(x, 1))
            
        # Fill NaN values
        result_df = result_df.fillna(0)
        
        logger.info(f"Extracted {len(result_df)} feature vectors for {chain_name}")
        return result_df
    
    except Exception as e:
        logger.error(f"Error extracting features from {chain_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def extract_network_features(all_addresses, chain_dir, chain_name):
    """Extract network-based features for addresses on a specific chain
    
    This function creates a graph of transactions and extracts network centrality measures
    as features for Sybil detection.
    
    Parameters:
    -----------
    all_addresses : pd.DataFrame
        DataFrame containing all addresses we're interested in
    chain_dir : str
        Directory containing the chain data
    chain_name : str
        Name of the chain (ethereum or base)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with network features for each address
    """
    try:
        logger.info(f"Extracting network features from {chain_name}...")
        
        # Create a lookup set for faster checks
        address_set = set(all_addresses['ADDRESS'].str.lower())
        
        # Initialize a directed graph
        G = nx.DiGraph()
        
        # Add all addresses as nodes
        for addr in address_set:
            G.add_node(addr)
        
        # Read transactions parquet
        tx_path = os.path.join(chain_dir, 'transactions.parquet')
        
        # Calculate batch size based on file size
        file_size_mb = os.path.getsize(tx_path) / (1024 * 1024)
        batch_size = min(100000, max(10000, int(200000000 / file_size_mb)))
        
        logger.info(f"Building transaction graph from {tx_path} in batches of {batch_size}")
        
        # Open parquet file
        pf = pq.ParquetFile(tx_path)
        
        # Track edges and weights
        edge_weights = defaultdict(int)
        edge_values = defaultdict(float)
        
        # Process in batches with tqdm progress bar
        total_batches = pf.num_row_groups
        for batch_i, batch in enumerate(tqdm(pf.iter_batches(batch_size=batch_size), 
                                          total=total_batches, 
                                          desc=f"Building {chain_name} transaction graph")):
            # Convert to pandas
            batch_df = batch.to_pandas()
            
            # Ensure addresses are lowercase
            batch_df['FROM_ADDRESS'] = batch_df['FROM_ADDRESS'].str.lower()
            batch_df['TO_ADDRESS'] = batch_df['TO_ADDRESS'].str.lower()
            
            # Process transactions to build graph
            for _, tx in batch_df.iterrows():
                from_addr = tx['FROM_ADDRESS']
                to_addr = tx['TO_ADDRESS']
                
                # Only add edges between addresses we're tracking
                if from_addr in address_set and to_addr in address_set:
                    # Update edge weight (number of transactions)
                    edge_weights[(from_addr, to_addr)] += 1
                    
                    # Update edge value (total value transferred), if available
                    if 'VALUE' in tx and pd.notna(tx['VALUE']):
                        value = float(tx['VALUE'])
                        if 0 <= value < 1e20:  # Filter extreme values
                            edge_values[(from_addr, to_addr)] += value
        
        # Add edges to graph
        for (from_addr, to_addr), weight in edge_weights.items():
            G.add_edge(from_addr, to_addr, weight=weight, value=edge_values.get((from_addr, to_addr), 0))
        
        logger.info(f"Network graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Extract network features
        logger.info("Calculating network centrality measures...")
        
        # Dictionary to store features
        features = {}
        
        # Degree centrality
        in_degree = dict(G.in_degree(weight='weight'))
        out_degree = dict(G.out_degree(weight='weight'))
        degree = {node: in_degree.get(node, 0) + out_degree.get(node, 0) for node in G.nodes()}
        
        # Page Rank centrality (limit iterations for speed)
        try:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, weight='weight')
        except:
            pagerank = {node: 0 for node in G.nodes()}
        
        # Betweenness centrality (approximate for speed)
        try:
            # Use approximate betweenness to improve speed
            betweenness = nx.approximation.betweenness_centrality(G, k=min(500, G.number_of_nodes()))
        except:
            betweenness = {node: 0 for node in G.nodes()}
        
        # Compute clustering coefficient
        try:
            clustering = nx.clustering(G.to_undirected())
        except:
            clustering = {node: 0 for node in G.nodes()}
        
        # Store all network features
        for addr in address_set:
            features[addr] = {
                'ADDRESS': addr,
                'in_degree': in_degree.get(addr, 0),
                'out_degree': out_degree.get(addr, 0),
                'total_degree': degree.get(addr, 0),
                'pagerank': pagerank.get(addr, 0),
                'betweenness': betweenness.get(addr, 0),
                'clustering': clustering.get(addr, 0),
            }
        
        # Create dataframe from features
        result_df = pd.DataFrame(list(features.values()))
        
        # Fill NaN values
        result_df = result_df.fillna(0)
        
        logger.info(f"Extracted network features for {len(result_df)} addresses")
        return result_df
    
    except Exception as e:
        logger.error(f"Error extracting network features from {chain_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def train_models(eth_features, base_features, all_train, test_features=None):
    """Train and evaluate chain-specific and combined models
    
    Parameters:
    -----------
    eth_features : pd.DataFrame
        Features extracted from Ethereum chain
    base_features : pd.DataFrame
        Features extracted from Base chain
    all_train : pd.DataFrame
        Training data with labels
    test_features : pd.DataFrame, optional
        Test data features for prediction
    
    Returns:
    --------
    dict
        Dictionary containing the trained models and predictions
    """
    try:
        logger.info("Training models...")
        
        # Merge features with training labels
        eth_train = eth_features.merge(all_train[['ADDRESS', 'LABEL']], on='ADDRESS', how='inner')
        base_train = base_features.merge(all_train[['ADDRESS', 'LABEL']], on='ADDRESS', how='inner')
        
        # Only keep training addresses
        eth_train_feats = eth_train.drop(columns=['ADDRESS', 'LABEL', 'chain'])
        base_train_feats = base_train.drop(columns=['ADDRESS', 'LABEL', 'chain'])
        
        # Get labels
        eth_train_labels = eth_train['LABEL']
        base_train_labels = base_train['LABEL']
        
        # Log feature counts
        logger.info(f"Ethereum training set: {len(eth_train_feats)} samples, {eth_train_feats.shape[1]} features")
        logger.info(f"Base training set: {len(base_train_feats)} samples, {base_train_feats.shape[1]} features")
        
        # Split into train/val for each chain
        eth_X_train, eth_X_val, eth_y_train, eth_y_val = train_test_split(
            eth_train_feats, eth_train_labels, test_size=0.2, random_state=SEED, stratify=eth_train_labels
        )
        
        base_X_train, base_X_val, base_y_train, base_y_val = train_test_split(
            base_train_feats, base_train_labels, test_size=0.2, random_state=SEED, stratify=base_train_labels
        )
        
        # Train Ethereum model
        logger.info("Training Ethereum model...")
        eth_model = train_chain_model(eth_X_train, eth_y_train, eth_X_val, eth_y_val, chain_name="ethereum")
        
        # Train Base model
        logger.info("Training Base model...")
        base_model = train_chain_model(base_X_train, base_y_train, base_X_val, base_y_val, chain_name="base")
        
        # Create combined features for addresses that exist in both chains
        logger.info("Creating combined features...")
        combined_features = create_combined_features(eth_features, base_features, all_train)
        
        # Split the combined features
        combined_train = combined_features.merge(all_train[['ADDRESS', 'LABEL']], on='ADDRESS', how='inner')
        combined_train_feats = combined_train.drop(columns=['ADDRESS', 'LABEL'])
        combined_train_labels = combined_train['LABEL']
        
        logger.info(f"Combined training set: {len(combined_train_feats)} samples, {combined_train_feats.shape[1]} features")
        
        # Split into train/val for combined model
        combined_X_train, combined_X_val, combined_y_train, combined_y_val = train_test_split(
            combined_train_feats, combined_train_labels, test_size=0.2, random_state=SEED, stratify=combined_train_labels
        )
        
        # Train combined model
        logger.info("Training combined model...")
        combined_model = train_chain_model(combined_X_train, combined_y_train, combined_X_val, combined_y_val, chain_name="combined")
        
        # Initialize predictions dict
        predictions = {}
        
        # If test features are provided, make predictions
        if test_features is not None:
            logger.info("Making predictions on test data...")
            
            # Get test addresses
            test_addresses = test_features['ADDRESS'].tolist()
            
            # Make predictions with each model
            eth_test_feats = eth_features[eth_features['ADDRESS'].isin(test_addresses)]
            base_test_feats = base_features[base_features['ADDRESS'].isin(test_addresses)]
            
            # Create combined test features
            combined_test_feats = create_combined_features(eth_features, base_features, test_features=test_features)
            
            # Predict with Ethereum model
            eth_preds = predict_with_model(eth_model, eth_test_feats, chain_name="ethereum")
            
            # Predict with Base model
            base_preds = predict_with_model(base_model, base_test_feats, chain_name="base")
            
            # Predict with combined model
            combined_preds = predict_with_model(combined_model, combined_test_feats, chain_name="combined")
            
            # Combine predictions using weighted average
            final_preds = combine_predictions(eth_preds, base_preds, combined_preds)
            
            # Store predictions
            predictions['eth_preds'] = eth_preds
            predictions['base_preds'] = base_preds
            predictions['combined_preds'] = combined_preds
            predictions['final_preds'] = final_preds
        
        # Store models
        models = {
            'eth_model': eth_model,
            'base_model': base_model,
            'combined_model': combined_model,
        }
        
        logger.info("Model training complete")
        return {'models': models, 'predictions': predictions}
    
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {'models': {}, 'predictions': {}}


def train_chain_model(X_train, y_train, X_val, y_val, chain_name):
    """Train a model for a specific blockchain
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation labels
    chain_name : str
        Name of the chain (ethereum, base, or combined)
        
    Returns:
    --------
    dict
        Dictionary containing the trained model and other metadata
    """
    try:
        # Check if we need to apply SMOTE for class imbalance
        class_counts = np.bincount(y_train)
        # Only apply SMOTE if we have a significant class imbalance and enough samples
        if len(class_counts) > 1 and min(class_counts) >= 5 and min(class_counts) / max(class_counts) < 0.3:
            logger.info(f"Applying SMOTE to handle class imbalance for {chain_name} model")
            smote = SMOTE(random_state=SEED)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_val_scaled = scaler.transform(X_val)
        
        # Train base models
        base_models = []
        base_preds = []
        
        # LightGBM model
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=8, 
            num_leaves=31,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=SEED
        )
        # Simplified fit without additional parameters
        lgb_model.fit(X_train_scaled, y_train_resampled)
        lgb_pred = lgb_model.predict_proba(X_val_scaled)[:, 1]
        base_models.append(('lgb', lgb_model))
        base_preds.append(lgb_pred)
        lgb_auc = roc_auc_score(y_val, lgb_pred)
        logger.info(f"{chain_name} LightGBM AUC: {lgb_auc:.4f}")
        
        # XGBoost model
        xgb_model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=SEED
        )
        # Simplified fit without additional parameters
        xgb_model.fit(X_train_scaled, y_train_resampled)
        xgb_pred = xgb_model.predict_proba(X_val_scaled)[:, 1]
        base_models.append(('xgb', xgb_model))
        base_preds.append(xgb_pred)
        xgb_auc = roc_auc_score(y_val, xgb_pred)
        logger.info(f"{chain_name} XGBoost AUC: {xgb_auc:.4f}")
        
        # Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=SEED,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train_resampled)
        rf_pred = rf_model.predict_proba(X_val_scaled)[:, 1]
        base_models.append(('rf', rf_model))
        base_preds.append(rf_pred)
        rf_auc = roc_auc_score(y_val, rf_pred)
        logger.info(f"{chain_name} Random Forest AUC: {rf_auc:.4f}")
        
        # Stack predictions for meta-model
        stacked_preds = np.column_stack(base_preds)
        
        # Train meta-model (logistic regression)
        meta_model = LogisticRegression(C=0.1, solver='liblinear', random_state=SEED)
        meta_model.fit(stacked_preds, y_val)
        meta_pred = meta_model.predict_proba(stacked_preds)[:, 1]
        meta_auc = roc_auc_score(y_val, meta_pred)
        logger.info(f"{chain_name} Meta-model AUC: {meta_auc:.4f}")
        
        # Save model components
        model_dict = {
            'base_models': base_models,
            'meta_model': meta_model,
            'scaler': scaler,
            'features': X_train.columns.tolist(),
            'auc': meta_auc
        }
        
        return model_dict
    
    except Exception as e:
        logger.error(f"Error training chain model for {chain_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_combined_features(eth_features, base_features, all_train=None, test_features=None):
    """Create combined features from both chains
    
    Parameters:
    -----------
    eth_features : pd.DataFrame
        Features from Ethereum chain
    base_features : pd.DataFrame
        Features from Base chain
    all_train : pd.DataFrame, optional
        Training data with labels
    test_features : pd.DataFrame, optional
        Test data features
        
    Returns:
    --------
    pd.DataFrame
        Combined features dataframe
    """
    try:
        # Determine which addresses to use
        if all_train is not None:
            addresses = all_train['ADDRESS'].tolist()
        elif test_features is not None:
            addresses = test_features['ADDRESS'].tolist()
        else:
            # Use intersection of addresses from both chains
            eth_addrs = set(eth_features['ADDRESS'])
            base_addrs = set(base_features['ADDRESS'])
            addresses = list(eth_addrs.intersection(base_addrs))
        
        # Create combined features
        results = []
        
        # Process each address
        for addr in addresses:
            row = {'ADDRESS': addr}
            
            # Extract Ethereum features if available
            eth_row = eth_features[eth_features['ADDRESS'] == addr]
            if not eth_row.empty:
                # Prefix Ethereum features with eth_
                for col in eth_row.columns:
                    if col not in ['ADDRESS', 'chain']:
                        row[f'eth_{col}'] = eth_row[col].values[0]
            else:
                # Fill with zeros if address not in Ethereum
                for col in eth_features.columns:
                    if col not in ['ADDRESS', 'chain']:
                        row[f'eth_{col}'] = 0
            
            # Extract Base features if available
            base_row = base_features[base_features['ADDRESS'] == addr]
            if not base_row.empty:
                # Prefix Base features with base_
                for col in base_row.columns:
                    if col not in ['ADDRESS', 'chain']:
                        row[f'base_{col}'] = base_row[col].values[0]
            else:
                # Fill with zeros if address not in Base
                for col in base_features.columns:
                    if col not in ['ADDRESS', 'chain']:
                        row[f'base_{col}'] = 0
            
            # Cross-chain features
            row['has_both_chains'] = 1 if not eth_row.empty and not base_row.empty else 0
            
            # Add common features
            if not eth_row.empty and not base_row.empty:
                # Transaction activity comparison
                if 'tx_count' in eth_row.columns and 'tx_count' in base_row.columns:
                    row['tx_count_ratio'] = eth_row['tx_count'].values[0] / max(1, base_row['tx_count'].values[0])
                
                # Value comparison
                if 'tx_value_sum' in eth_row.columns and 'tx_value_sum' in base_row.columns:
                    row['value_sum_ratio'] = eth_row['tx_value_sum'].values[0] / max(1, base_row['tx_value_sum'].values[0])
                
                # Token activity comparison
                if 'token_tx_count' in eth_row.columns and 'token_tx_count' in base_row.columns:
                    row['token_tx_ratio'] = eth_row['token_tx_count'].values[0] / max(1, base_row['token_tx_count'].values[0])
            
            results.append(row)
        
        # Create dataframe
        combined_df = pd.DataFrame(results)
        
        # Fill NaN values
        combined_df = combined_df.fillna(0)
        
        return combined_df
    
    except Exception as e:
        logger.error(f"Error creating combined features: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def predict_with_model(model_dict, test_features, chain_name):
    """Make predictions using a trained model
    
    Parameters:
    -----------
    model_dict : dict
        Dictionary containing the trained model and metadata
    test_features : pd.DataFrame
        Test features to predict on
    chain_name : str
        Name of the chain
        
    Returns:
    --------
    dict
        Dictionary mapping addresses to predicted probabilities
    """
    try:
        if model_dict is None:
            logger.error(f"Model for {chain_name} is None, cannot make predictions")
            return {}
        
        # Extract components
        base_models = model_dict['base_models']
        meta_model = model_dict['meta_model']
        scaler = model_dict['scaler']
        model_features = model_dict['features']
        
        # Get addresses
        addresses = test_features['ADDRESS'].tolist()
        
        # Extract features
        X_test = test_features.drop(columns=['ADDRESS', 'chain'])
        
        # Ensure features match the model's expected features
        missing_cols = set(model_features) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        X_test = X_test[model_features]
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions with base models
        base_preds = []
        for name, model in base_models:
            pred = model.predict_proba(X_test_scaled)[:, 1]
            base_preds.append(pred)
        
        # Stack predictions
        stacked_preds = np.column_stack(base_preds)
        
        # Make predictions with meta-model
        final_preds = meta_model.predict_proba(stacked_preds)[:, 1]
        
        # Map predictions to addresses
        predictions = {addr: pred for addr, pred in zip(addresses, final_preds)}
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error predicting with {chain_name} model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def combine_predictions(eth_preds, base_preds, combined_preds):
    """Combine predictions from different models
    
    Parameters:
    -----------
    eth_preds : dict
        Predictions from Ethereum model
    base_preds : dict
        Predictions from Base model
    combined_preds : dict
        Predictions from combined model
        
    Returns:
    --------
    dict
        Dictionary mapping addresses to final probabilities
    """
    try:
        # Get all unique addresses
        all_addresses = set(eth_preds.keys()).union(set(base_preds.keys())).union(set(combined_preds.keys()))
        
        # Combine predictions using weighted average
        final_preds = {}
        
        for addr in all_addresses:
            eth_score = eth_preds.get(addr, 0)
            base_score = base_preds.get(addr, 0)
            combined_score = combined_preds.get(addr, 0)
            
            # Weighted average
            final_score = (eth_score * ETH_WEIGHT + 
                          base_score * BASE_WEIGHT + 
                          combined_score * COMBINED_WEIGHT)
            
            final_preds[addr] = final_score
        
        return final_preds
    
    except Exception as e:
        logger.error(f"Error combining predictions: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


# Main function to run the entire pipeline
def main():
    try:
        logger.info("Starting hybrid ensemble model pipeline...")
        logger.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load addresses
        all_train, all_test, eth_train, eth_test, base_train, base_test, eth_addresses, base_addresses = load_addresses()
        
        if all_train is None or all_test is None:
            logger.error("Failed to load addresses, exiting")
            return
        
        # Check if features are cached
        use_cache = os.path.exists(cache_eth_features) and os.path.exists(cache_base_features)
        
        # Extract features
        if use_cache:
            logger.info("Loading features from cache...")
            with open(cache_eth_features, 'rb') as f:
                eth_features = pickle.load(f)
            with open(cache_base_features, 'rb') as f:
                base_features = pickle.load(f)
        else:
            logger.info("Extracting features from scratch...")
            
            # Extract Ethereum features
            eth_features = extract_basic_features(ETHEREUM_DIR, eth_addresses, "ethereum")
            
            # Extract Base features
            base_features = extract_basic_features(BASE_DIR, base_addresses, "base")
            
            # Extract network features if we have enough addresses
            if len(eth_addresses) > 50:
                eth_network = extract_network_features(eth_addresses, ETHEREUM_DIR, "ethereum")
                eth_features = eth_features.merge(eth_network, on='ADDRESS', how='left')
            
            if len(base_addresses) > 50:
                base_network = extract_network_features(base_addresses, BASE_DIR, "base")
                base_features = base_features.merge(base_network, on='ADDRESS', how='left')
            
            # Fill missing values
            eth_features = eth_features.fillna(0)
            base_features = base_features.fillna(0)
            
            # Cache features
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_eth_features, 'wb') as f:
                pickle.dump(eth_features, f)
            with open(cache_base_features, 'wb') as f:
                pickle.dump(base_features, f)
        
        # Train models and make predictions
        result = train_models(eth_features, base_features, all_train, all_test)
        
        if not result['predictions']:
            logger.error("No predictions generated, exiting")
            return
        
        # Save trained models
        logger.info("Saving trained models...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(f"{OUTPUT_DIR}/eth_model.pkl", 'wb') as f:
            pickle.dump(result['models']['eth_model'], f)
        with open(f"{OUTPUT_DIR}/base_model.pkl", 'wb') as f:
            pickle.dump(result['models']['base_model'], f)
        with open(f"{OUTPUT_DIR}/combined_model.pkl", 'wb') as f:
            pickle.dump(result['models']['combined_model'], f)
        
        logger.info(f"✅ Training completed in {(time.time() - start_time) / 60:.2f} minutes")
        logger.info(f"✅ Models saved to {OUTPUT_DIR}/")
        logger.info(f"✅ Ready to use with predict_pond.py")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# Run the main function if script is executed directly
if __name__ == "__main__":
    main()
