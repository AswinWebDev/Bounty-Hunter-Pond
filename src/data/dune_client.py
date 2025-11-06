"""
Dune Analytics API Client

Fetch blockchain data using Dune API (free tier available).
Bypasses need for premium CSV exports.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests


def load_dune_api_key() -> Optional[str]:
    """Load Dune API key from api.txt"""
    api_file = Path(__file__).resolve().parents[2] / "api.txt"
    
    if not api_file.exists():
        return None
    
    with open(api_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('dune_api_key='):
                return line.split('=', 1)[1].strip()
    
    return None


class DuneClient:
    """
    Dune Analytics API client
    
    Free tier: 300 credits/day
    Each query execution costs ~20 credits
    = ~15 queries per day on free tier
    
    API docs: https://docs.dune.com/api-reference/
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or load_dune_api_key()
        if not self.api_key:
            raise ValueError("Dune API key not found. Add 'dune_api_key=YOUR_KEY' to api.txt")
        
        self.base_url = "https://api.dune.com/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "X-Dune-API-Key": self.api_key,
            "Content-Type": "application/json"
        })
    
    def execute_query(self, query_id: int, params: Optional[Dict] = None) -> str:
        """
        Execute a Dune query and return execution ID
        
        Args:
            query_id: Dune query ID (from URL)
            params: Optional query parameters
        
        Returns:
            execution_id: ID to check results
        """
        url = f"{self.base_url}/query/{query_id}/execute"
        
        payload = {}
        if params:
            payload["query_parameters"] = params
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        execution_id = data.get("execution_id")
        
        print(f"✓ Query {query_id} started. Execution ID: {execution_id}")
        return execution_id
    
    def get_execution_status(self, execution_id: str) -> Dict:
        """Check status of query execution"""
        url = f"{self.base_url}/execution/{execution_id}/status"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def get_execution_results(self, execution_id: str) -> pd.DataFrame:
        """
        Get results of completed query execution
        
        Args:
            execution_id: Execution ID from execute_query
        
        Returns:
            DataFrame with query results
        """
        url = f"{self.base_url}/execution/{execution_id}/results"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        data = response.json()
        rows = data.get("result", {}).get("rows", [])
        
        if not rows:
            print("⚠️  Query returned no results")
            return pd.DataFrame()
        
        return pd.DataFrame(rows)
    
    def run_query(self, query_id: int, params: Optional[Dict] = None, max_wait: int = 300) -> pd.DataFrame:
        """
        Execute query and wait for results
        
        Args:
            query_id: Dune query ID
            params: Optional parameters
            max_wait: Maximum seconds to wait for completion
        
        Returns:
            DataFrame with results
        """
        print(f"Executing Dune query {query_id}...")
        
        # Start execution
        execution_id = self.execute_query(query_id, params)
        
        # Poll for completion
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status_data = self.get_execution_status(execution_id)
            state = status_data.get("state")
            
            if state == "QUERY_STATE_COMPLETED":
                print(f"✓ Query completed in {time.time() - start_time:.1f}s")
                return self.get_execution_results(execution_id)
            elif state == "QUERY_STATE_FAILED":
                error = status_data.get("error")
                raise RuntimeError(f"Query failed: {error}")
            
            print(f"  Status: {state}, waiting...")
            time.sleep(5)
        
        raise TimeoutError(f"Query did not complete within {max_wait}s")
    
    def run_custom_query(self, sql: str, max_wait: int = 300) -> pd.DataFrame:
        """
        Run a custom SQL query (requires creating query on Dune first)
        
        For now, this is a helper to create queries programmatically.
        You'll need to:
        1. Go to dune.com
        2. Create query with this SQL
        3. Get query ID from URL
        4. Use run_query(query_id)
        """
        print("⚠️  Custom SQL execution requires creating query on Dune website first")
        print("Steps:")
        print("1. Go to https://dune.com/queries")
        print("2. Click 'New Query'")
        print("3. Paste this SQL:")
        print("-" * 60)
        print(sql)
        print("-" * 60)
        print("4. Save query and get ID from URL (e.g., dune.com/queries/1234567)")
        print("5. Run: dune.run_query(query_id=1234567)")
        
        return None


# Pre-made queries for common tasks
class DuneQueries:
    """Helper class with common Dune query templates"""
    
    @staticmethod
    def base_transactions_recent(days: int = 30, limit: int = 100000) -> str:
        """Get recent Base transactions"""
        return f"""
        SELECT 
            "from" as from_address,
            "to" as to_address,
            value,
            gas_price,
            gas_used,
            block_time as block_timestamp,
            hash as tx_hash
        FROM base.transactions
        WHERE block_time >= NOW() - INTERVAL '{days}' DAY
        ORDER BY block_time DESC
        LIMIT {limit}
        """
    
    @staticmethod
    def base_addresses_activity(addresses: List[str], limit: int = 10) -> str:
        """Get recent activity for specific addresses"""
        addr_list = "', '".join([a.lower() for a in addresses])
        return f"""
        SELECT 
            "from" as from_address,
            "to" as to_address,
            value,
            gas_price,
            gas_used,
            block_time as block_timestamp,
            hash as tx_hash
        FROM base.transactions
        WHERE ("from" IN ('{addr_list}') OR "to" IN ('{addr_list}'))
        ORDER BY block_time DESC
        LIMIT {limit}
        """
    
    @staticmethod
    def base_token_transfers(addresses: List[str], days: int = 30) -> str:
        """Get token transfers for addresses"""
        addr_list = "', '".join([a.lower() for a in addresses])
        return f"""
        SELECT 
            "from" as from_address,
            "to" as to_address,
            contract_address,
            value,
            block_time as block_timestamp
        FROM erc20_base.evt_Transfer
        WHERE ("from" IN ('{addr_list}') OR "to" IN ('{addr_list}'))
            AND block_time >= NOW() - INTERVAL '{days}' DAY
        ORDER BY block_time DESC
        LIMIT 50000
        """


if __name__ == '__main__':
    # Test Dune API connection
    try:
        client = DuneClient()
        print(f"✓ Dune API key loaded")
        print("\nTo use:")
        print("1. Create query on dune.com with desired SQL")
        print("2. Get query ID from URL")
        print("3. Run: client.run_query(query_id)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nAdd your Dune API key to api.txt:")
        print("dune_api_key=YOUR_API_KEY_HERE")
