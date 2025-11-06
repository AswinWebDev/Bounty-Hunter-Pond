from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm

from src.data.explorer_client import EtherscanLikeClient


def fetch_address_activity(addresses: Iterable[str], chains: Iterable[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch txlist and tokentx for each address for each chain.
    Returns nested dict: {chain: {"tx": df, "tokentx": df}}
    """
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    addr_list = list(addresses)
    for chain in chains:
        client = EtherscanLikeClient.for_chain(chain)
        all_tx_rows: List[dict] = []
        all_tok_rows: List[dict] = []
        for addr in tqdm(addr_list, desc=f"{chain}: addresses", ncols=80):
            try:
                txs = client.account_txlist(addr)
                # API may return error string instead of list
                if not isinstance(txs, list):
                    txs = []
            except Exception:
                txs = []
            for r in (txs or []):
                if isinstance(r, dict):
                    r = dict(r)
                    r["chain"] = chain
                    r["address_focus"] = addr
                    all_tx_rows.append(r)
            try:
                toks = client.account_tokentx(addr)
                if not isinstance(toks, list):
                    toks = []
            except Exception:
                toks = []
            for r in (toks or []):
                if isinstance(r, dict):
                    r = dict(r)
                    r["chain"] = chain
                    r["address_focus"] = addr
                    all_tok_rows.append(r)
        tx_df = pd.DataFrame(all_tx_rows) if all_tx_rows else pd.DataFrame()
        tok_df = pd.DataFrame(all_tok_rows) if all_tok_rows else pd.DataFrame()
        out[chain] = {"tx": tx_df, "tokentx": tok_df}
    return out


to_int_cols = [
    "timeStamp",
    "value",
    "gas",
    "gasPrice",
    "nonce",
    "blockNumber",
]


def _coerce_txs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    for c in to_int_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "input" in d.columns:
        d["method_sig"] = d["input"].astype(str).str[:10]
    if "from" in d.columns:
        d["from"] = d["from"].astype(str).str.lower()
    if "to" in d.columns:
        d["to"] = d["to"].astype(str).str.lower()
    return d


def _coerce_tok(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    for c in ["value", "timeStamp", "blockNumber"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["from", "to", "contractAddress"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.lower()
    return d


def save_activity(activity: Dict[str, Dict[str, pd.DataFrame]], out_dir: Path) -> Dict[str, Dict[str, Path]]:
    out_paths: Dict[str, Dict[str, Path]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for chain, parts in activity.items():
        tx_df = _coerce_txs(parts.get("tx", pd.DataFrame()))
        tok_df = _coerce_tok(parts.get("tokentx", pd.DataFrame()))
        chain_dir = out_dir / chain
        chain_dir.mkdir(parents=True, exist_ok=True)
        tx_p = chain_dir / "tx.csv"
        tok_p = chain_dir / "tokentx.csv"
        if not tx_df.empty:
            tx_df.to_csv(tx_p, index=False)
        if not tok_df.empty:
            tok_df.to_csv(tok_p, index=False)
        out_paths[chain] = {"tx": tx_p, "tokentx": tok_p}
    return out_paths
