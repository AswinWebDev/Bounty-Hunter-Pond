"""
Weak label functions for Sybil/Bot detection

Each function returns a tuple: (vote, confidence, reason)
- vote: +1 (sybil/bot), -1 (legit), 0 (abstain)
- confidence: 0.0-1.0
- reason: string explanation
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class LabelResult:
    address: str
    label: str  # "sybil", "legit", "unknown"
    confidence: float
    reasons: List[str]
    votes: Dict[str, Tuple[int, float]]  # function_name -> (vote, conf)


def lf_burst_of_life(tx_df: pd.DataFrame, address: str) -> Tuple[int, float, str]:
    """Detect burst-of-life: many tx in short window, then dormant"""
    addr_tx = tx_df[tx_df["address_focus"] == address]
    if len(addr_tx) < 5:
        return (0, 0.0, "insufficient_tx")
    
    times = pd.to_numeric(addr_tx["timeStamp"], errors="coerce").dropna().sort_values()
    if len(times) < 5:
        return (0, 0.0, "insufficient_timestamps")
    
    # Find max tx in 6-hour window
    window = 6 * 3600
    max_in_window = 0
    for i in range(len(times)):
        start = times.iloc[i]
        end = start + window
        count = ((times >= start) & (times <= end)).sum()
        max_in_window = max(max_in_window, count)
    
    burst_ratio = max_in_window / len(times)
    lifespan = (times.max() - times.min()) / 86400  # days
    
    if burst_ratio >= 0.5 and lifespan > 1:
        conf = min(0.9, burst_ratio * 0.8 + 0.1)
        return (+1, conf, f"burst_ratio={burst_ratio:.2f}_in_6h")
    return (0, 0.0, "no_burst")


def lf_shared_funder(tx_df: pd.DataFrame, address: str) -> Tuple[int, float, str]:
    """Detect if >80% of inbound tx from single funder"""
    addr_tx = tx_df[tx_df["address_focus"] == address]
    inbound = addr_tx[addr_tx["to"].astype(str).str.lower() == address.lower()]
    
    if len(inbound) < 4:
        return (0, 0.0, "insufficient_inbound")
    
    from_addrs = inbound["from"].value_counts(normalize=True)
    if len(from_addrs) == 0:
        return (0, 0.0, "no_funders")
    
    top_ratio = from_addrs.iloc[0]
    if top_ratio >= 0.8:
        conf = min(0.9, top_ratio * 0.9)
        return (+1, conf, f"shared_funder={top_ratio:.2f}")
    return (0, 0.0, "diverse_funders")


def lf_low_counterparty_diversity(tx_df: pd.DataFrame, address: str) -> Tuple[int, float, str]:
    """Detect low diversity in recipients"""
    addr_tx = tx_df[tx_df["address_focus"] == address]
    outbound = addr_tx[addr_tx["from"].astype(str).str.lower() == address.lower()]
    
    if len(outbound) < 5:
        return (0, 0.0, "insufficient_outbound")
    
    unique_to = outbound["to"].nunique()
    diversity = unique_to / len(outbound)
    
    if unique_to <= 2:
        conf = 0.7
        return (+1, conf, f"unique_to={unique_to}")
    elif diversity < 0.3:
        conf = 0.5
        return (+1, conf, f"diversity={diversity:.2f}")
    return (0, 0.0, "diverse_counterparties")


def lf_method_template(tx_df: pd.DataFrame, address: str) -> Tuple[int, float, str]:
    """Detect repeated method signatures (template behavior)"""
    addr_tx = tx_df[tx_df["address_focus"] == address]
    outbound = addr_tx[addr_tx["from"].astype(str).str.lower() == address.lower()]
    
    if len(outbound) < 5:
        return (0, 0.0, "insufficient_outbound")
    
    if "method_sig" not in outbound.columns:
        return (0, 0.0, "no_method_sig")
    
    methods = outbound["method_sig"].value_counts(normalize=True)
    if len(methods) == 0:
        return (0, 0.0, "no_methods")
    
    top_method_ratio = methods.iloc[0]
    if top_method_ratio >= 0.8 and len(methods) <= 2:
        conf = min(0.8, top_method_ratio * 0.85)
        return (+1, conf, f"method_template={top_method_ratio:.2f}")
    return (0, 0.0, "varied_methods")


def lf_gas_template(tx_df: pd.DataFrame, address: str) -> Tuple[int, float, str]:
    """Detect identical gas prices (automated behavior)"""
    addr_tx = tx_df[tx_df["address_focus"] == address]
    outbound = addr_tx[addr_tx["from"].astype(str).str.lower() == address.lower()]
    
    if len(outbound) < 5:
        return (0, 0.0, "insufficient_outbound")
    
    gas_prices = pd.to_numeric(outbound.get("gasPrice", pd.Series()), errors="coerce").dropna()
    if len(gas_prices) < 5:
        return (0, 0.0, "no_gas_prices")
    
    unique_gas = gas_prices.nunique()
    if unique_gas == 1:
        return (+1, 0.75, f"identical_gas={unique_gas}")
    elif unique_gas <= 2:
        return (+1, 0.5, f"gas_template={unique_gas}")
    return (0, 0.0, "varied_gas")


def lf_token_monoculture(tok_df: pd.DataFrame, address: str) -> Tuple[int, float, str]:
    """Detect single-token focus (farming behavior)"""
    addr_tok = tok_df[tok_df["address_focus"] == address]
    if len(addr_tok) < 5:
        return (0, 0.0, "insufficient_token_tx")
    
    if "contractAddress" not in addr_tok.columns:
        return (0, 0.0, "no_contract_address")
    
    contracts = addr_tok["contractAddress"].value_counts(normalize=True)
    if len(contracts) == 0:
        return (0, 0.0, "no_tokens")
    
    top_contract_ratio = contracts.iloc[0]
    if top_contract_ratio >= 0.9:
        conf = min(0.8, top_contract_ratio * 0.85)
        return (+1, conf, f"token_monoculture={top_contract_ratio:.2f}")
    return (0, 0.0, "diverse_tokens")


def lf_long_lifespan(tx_df: pd.DataFrame, address: str) -> Tuple[int, float, str]:
    """Long active lifespan suggests legitimate use"""
    addr_tx = tx_df[tx_df["address_focus"] == address]
    if len(addr_tx) < 2:
        return (0, 0.0, "insufficient_tx")
    
    times = pd.to_numeric(addr_tx["timeStamp"], errors="coerce").dropna()
    if len(times) < 2:
        return (0, 0.0, "no_timestamps")
    
    lifespan_days = (times.max() - times.min()) / 86400
    if lifespan_days >= 90:
        conf = min(0.8, lifespan_days / 180 * 0.7 + 0.1)
        return (-1, conf, f"lifespan={lifespan_days:.0f}d")
    return (0, 0.0, "short_lifespan")


def lf_diverse_activity(tx_df: pd.DataFrame, address: str) -> Tuple[int, float, str]:
    """High counterparty and method diversity suggests legitimate use"""
    addr_tx = tx_df[tx_df["address_focus"] == address]
    if len(addr_tx) < 10:
        return (0, 0.0, "insufficient_tx")
    
    unique_to = addr_tx["to"].nunique()
    unique_from = addr_tx["from"].nunique()
    unique_methods = addr_tx.get("method_sig", pd.Series()).nunique()
    
    diversity_score = (unique_to + unique_from + unique_methods) / len(addr_tx)
    if diversity_score >= 0.5 and unique_to >= 10:
        conf = min(0.75, diversity_score * 0.7)
        return (-1, conf, f"diversity={diversity_score:.2f}")
    return (0, 0.0, "low_diversity")


LABEL_FUNCTIONS = [
    ("burst_of_life", lf_burst_of_life),
    ("shared_funder", lf_shared_funder),
    ("low_counterparty_diversity", lf_low_counterparty_diversity),
    ("method_template", lf_method_template),
    ("gas_template", lf_gas_template),
    ("token_monoculture", lf_token_monoculture),
    ("long_lifespan", lf_long_lifespan),
    ("diverse_activity", lf_diverse_activity),
]


def aggregate_votes(votes: Dict[str, Tuple[int, float]]) -> Tuple[str, float]:
    """
    Aggregate votes from multiple label functions
    Returns (label, confidence)
    """
    if not votes:
        return ("unknown", 0.0)
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for name, (vote, conf) in votes.items():
        if vote != 0:
            weighted_sum += vote * conf
            total_weight += conf
    
    if total_weight < 0.1:
        return ("unknown", 0.0)
    
    avg_vote = weighted_sum / total_weight
    confidence = min(0.95, total_weight / len(votes))
    
    if avg_vote >= 0.3:
        return ("sybil", confidence)
    elif avg_vote <= -0.3:
        return ("legit", confidence)
    else:
        return ("unknown", confidence * 0.5)


def apply_weak_labels(tx_df: pd.DataFrame, tok_df: pd.DataFrame, addresses: List[str]) -> pd.DataFrame:
    """
    Apply all label functions to addresses
    Returns DataFrame with columns: address, label, confidence, reasons
    """
    results = []
    for addr in addresses:
        votes = {}
        reasons = []
        
        for name, func in LABEL_FUNCTIONS:
            if name.startswith("token") and tok_df is not None:
                vote, conf, reason = func(tok_df, addr)
            else:
                vote, conf, reason = func(tx_df, addr)
            
            votes[name] = (vote, conf)
            if vote != 0 and conf > 0.3:
                prefix = "+" if vote > 0 else "-"
                reasons.append(f"{prefix}{name}({reason})")
        
        label, confidence = aggregate_votes(votes)
        results.append({
            "address": addr,
            "label": label,
            "confidence": confidence,
            "reasons": "; ".join(reasons) if reasons else "no_signals",
            "n_votes": sum(1 for v, c in votes.values() if v != 0),
        })
    
    return pd.DataFrame(results)
