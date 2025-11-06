from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


def load_api_key() -> str:
    """Load Etherscan API key from .env or api.txt (in that order)"""
    # Try .env first (best practice)
    root_dir = Path(__file__).resolve().parent.parent.parent
    env_file = root_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    api_key = os.getenv("ETHERSCAN_API_KEY")
    if api_key and api_key != "your_api_key_here":
        return api_key
    
    # Fall back to api.txt (legacy support)
    api_file = root_dir / "api.txt"
    if api_file.exists():
        for line in api_file.read_text().splitlines():
            if line.strip().startswith("etherscan_api_key="):
                return line.split("=", 1)[1].strip()
    
    return ""


def explorer_base_url() -> str:
    """Return base URL for BaseScan API"""
    return "https://api.basescan.org/api"


class EtherscanLikeClient:
    """
    Minimal Etherscan-compatible client using the `proxy` module for JSON-RPC equivalents.
    Defaults to BaseScan endpoint.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, min_interval_s: float = 0.2, chain_id: Optional[int] = None) -> None:
        self.api_key = api_key or load_api_key()
        if not self.api_key:
            raise RuntimeError("ETHERSCAN_API_KEY not found. Set env var or put etherscan_api_key=... in api.txt")
        self.base_url = base_url or explorer_base_url()
        self.chain_id = chain_id  # For V2 API
        self._last_call = 0.0
        self._min_interval_s = min_interval_s
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "pond-bounty-bot-detector/0.1"})

    def _throttle(self) -> None:
        now = time.time()
        delta = now - self._last_call
        if delta < self._min_interval_s:
            time.sleep(self._min_interval_s - delta)
        self._last_call = time.time()

    def _call(self, module: str, action: str, **params: Any) -> Any:
        q: Dict[str, Any] = {"module": module, "action": action, "apikey": self.api_key}
        # Add chainid for V2 API
        if self.chain_id:
            q["chainid"] = self.chain_id
        q.update(params)
        self._throttle()
        r = self._session.get(self.base_url, params=q, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "result" not in data:
            raise RuntimeError(f"Explorer response missing result: {data}")
        return data["result"]

    @staticmethod
    def _to_hex_tag(block_number: int) -> str:
        return hex(block_number)

    def eth_block_number(self) -> int:
        # Use block module instead of deprecated proxy
        try:
            res = self._call("block", "getblocknumber")
            return int(res)
        except Exception:
            # Fallback to proxy if block module unavailable
            res = self._call("proxy", "eth_blockNumber")
            return int(res, 16)

    def get_transactions(self, address: str, limit: int = 10) -> list:
        """Get recent transactions for an address"""
        try:
            # Get normal transactions
            result = self._call(
                "account",
                "txlist",
                address=address,
                startblock=0,
                endblock=99999999,
                page=1,
                offset=limit,
                sort="desc"
            )
            
            if isinstance(result, str) and result.lower() == "no transactions found":
                return []
            
            if not isinstance(result, list):
                return []
            
            return result
        except Exception as e:
            print(f"Error fetching transactions: {e}")
            return []
    
    def eth_get_block_by_number(self, block_number: int, full_tx: bool = True) -> Dict[str, Any]:
        tag = self._to_hex_tag(block_number)
        boolean = "true" if full_tx else "false"
        res = self._call("proxy", "eth_getBlockByNumber", tag=tag, boolean=boolean)
        return res

    # --- Account module helpers (address-targeted ingestion) ---
    def account_txlist(self, address: str, startblock: int = 0, endblock: int = 99999999, page: int = 1, offset: int = 10000, sort: str = "asc") -> Any:
        return self._call(
            "account",
            "txlist",
            address=address,
            startblock=startblock,
            endblock=endblock,
            page=page,
            offset=offset,
            sort=sort,
        )

    def account_txlistinternal(self, address: str, startblock: int = 0, endblock: int = 99999999, page: int = 1, offset: int = 10000, sort: str = "asc") -> Any:
        return self._call(
            "account",
            "txlistinternal",
            address=address,
            startblock=startblock,
            endblock=endblock,
            page=page,
            offset=offset,
            sort=sort,
        )

    def account_tokentx(self, address: str, startblock: int = 0, endblock: int = 99999999, page: int = 1, offset: int = 10000, sort: str = "asc") -> Any:
        return self._call(
            "account",
            "tokentx",
            address=address,
            startblock=startblock,
            endblock=endblock,
            page=page,
            offset=offset,
            sort=sort,
        )

    def account_tokennfttx(self, address: str, startblock: int = 0, endblock: int = 99999999, page: int = 1, offset: int = 10000, sort: str = "asc") -> Any:
        return self._call(
            "account",
            "tokennfttx",
            address=address,
            startblock=startblock,
            endblock=endblock,
            page=page,
            offset=offset,
            sort=sort,
        )

    @classmethod
    def for_chain(cls, chain: str, api_key: Optional[str] = None, min_interval_s: float = 0.2) -> "EtherscanLikeClient":
        """
        Create client for a specific chain using Etherscan V2 API.
        V2 API works for all chains with a single endpoint and chain ID.
        """
        chain = chain.lower()
        # Use Etherscan V2 API (unified endpoint for all chains)
        base_url = "https://api.etherscan.io/v2/api"
        
        # Set chain ID for V2 API
        if chain in ("base", "basescan"):
            chain_id = 8453  # Base mainnet
        elif chain in ("eth", "ethereum", "etherscan"):
            chain_id = 1  # Ethereum mainnet
        else:
            raise ValueError(f"Unsupported chain: {chain}")
        
        return cls(api_key=api_key, base_url=base_url, min_interval_s=min_interval_s, chain_id=chain_id)
