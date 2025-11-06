#!/usr/bin/env python
"""
Social Profile Data Collector

Collects Twitter, GitHub, and LinkedIn data for addresses that have
social profiles linked on Pond platform.

Strategy:
- Use public APIs where possible (GitHub is free, Twitter has free tier)
- Scrape public profile pages as fallback
- Build social graph of connections
- Extract Sybil detection signals
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
from tqdm import tqdm
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CACHE_DIR = ROOT / "cache" / "social"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# GitHub Features (Free API)
# ==========================

def get_github_profile(username: str) -> Optional[Dict]:
    """Fetch GitHub profile using public API (no auth needed for public data)"""
    cache_path = CACHE_DIR / "github" / f"{username}.json"
    
    # Check cache
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    try:
        # Public GitHub API
        url = f"https://api.github.com/users/{username}"
        resp = requests.get(url, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            
            # Get repositories
            repos_url = f"https://api.github.com/users/{username}/repos"
            repos_resp = requests.get(repos_url, params={'per_page': 100}, timeout=10)
            repos = repos_resp.json() if repos_resp.status_code == 200 else []
            
            profile = {
                'username': username,
                'account_created': data.get('created_at'),
                'public_repos': data.get('public_repos', 0),
                'followers': data.get('followers', 0),
                'following': data.get('following', 0),
                'bio': data.get('bio', ''),
                'company': data.get('company', ''),
                'location': data.get('location', ''),
                'blog': data.get('blog', ''),
                'twitter_username': data.get('twitter_username', ''),
                'repos': repos,
                'total_stars': sum(r.get('stargazers_count', 0) for r in repos),
                'original_repos': sum(1 for r in repos if not r.get('fork', False)),
                'fetched_at': datetime.now().isoformat(),
            }
            
            # Save to cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            return profile
        else:
            print(f"GitHub API error for {username}: {resp.status_code}")
            return None
    
    except Exception as e:
        print(f"Error fetching GitHub profile for {username}: {e}")
        return None


def extract_github_features(profile: Dict) -> Dict:
    """Extract Sybil detection features from GitHub profile"""
    if not profile:
        return {
            'github_exists': False,
            'github_account_age_days': 0,
            'github_repos': 0,
            'github_followers': 0,
            'github_following': 0,
            'github_stars': 0,
            'github_original_repos': 0,
            'github_has_bio': False,
            'github_has_company': False,
            'github_follower_ratio': 0,
            'github_empty_profile': True,
            'github_new_account': True,
            'github_no_activity': True,
            'github_suspicious': True,
        }
    
    # Calculate account age
    created = pd.to_datetime(profile.get('account_created'))
    if created:
        created = created.tz_localize(None) if created.tzinfo else created
        account_age_days = (datetime.now() - created).days
    else:
        account_age_days = 0
    
    followers = profile.get('followers', 0)
    following = profile.get('following', 0)
    repos = profile.get('public_repos', 0)
    stars = profile.get('total_stars', 0)
    original_repos = profile.get('original_repos', 0)
    
    return {
        'github_exists': True,
        'github_username': profile.get('username'),
        'github_account_age_days': account_age_days,
        'github_repos': repos,
        'github_followers': followers,
        'github_following': following,
        'github_stars': stars,
        'github_original_repos': original_repos,
        'github_has_bio': len(profile.get('bio') or '') > 0,
        'github_has_company': len(profile.get('company') or '') > 0,
        'github_follower_ratio': followers / max(following, 1),
        
        # Sybil signals
        'github_empty_profile': repos == 0 and stars == 0,
        'github_new_account': account_age_days < 30,
        'github_no_activity': original_repos == 0,
        'github_suspicious': (repos == 0 or account_age_days < 30) and followers < 5,
    }


# ==========================
# Twitter Features
# ==========================

def get_twitter_profile_nitter(username: str) -> Optional[Dict]:
    """Fetch Twitter profile using Nitter (no API key needed)"""
    cache_path = CACHE_DIR / "twitter" / f"{username}.json"
    
    # Check cache
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    try:
        # Use public Nitter instance
        url = f"https://nitter.net/{username}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            # Parse HTML (basic scraping)
            html = resp.text
            
            # Extract basic stats (this is simplified - would need proper HTML parsing)
            # For now, just save the HTML and mark as found
            profile = {
                'username': username,
                'exists': True,
                'fetched_at': datetime.now().isoformat(),
                'html': html[:1000],  # Save first 1000 chars for debugging
            }
            
            # Save to cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            return profile
        else:
            return None
    
    except Exception as e:
        print(f"Error fetching Twitter profile for {username}: {e}")
        return None


def extract_twitter_features(profile: Dict) -> Dict:
    """Extract Twitter features (placeholder - needs proper implementation)"""
    if not profile or not profile.get('exists'):
        return {
            'twitter_exists': False,
            'twitter_suspicious': True,
        }
    
    # For now, just mark as exists
    # TODO: Implement proper HTML parsing or use Twitter API
    return {
        'twitter_exists': True,
        'twitter_username': profile.get('username'),
        'twitter_suspicious': False,  # Need data to determine
    }


# ==========================
# Cross-Platform Features
# ==========================

def extract_cross_platform_features(github_profile: Dict, twitter_profile: Dict) -> Dict:
    """Check consistency across platforms"""
    
    features = {}
    
    # Check if profiles are linked
    if github_profile and twitter_profile:
        github_twitter = (github_profile.get('twitter_username') or '').lower()
        twitter_username = (twitter_profile.get('username') or '').lower()
        
        features['profiles_linked'] = github_twitter == twitter_username and bool(github_twitter)
    else:
        features['profiles_linked'] = False
    
    # Check if both exist
    features['has_github'] = github_profile is not None
    features['has_twitter'] = twitter_profile is not None and twitter_profile.get('exists', False)
    features['multi_platform'] = features['has_github'] and features['has_twitter']
    
    # Sybil signal: only one platform
    features['single_platform_only'] = (features['has_github'] or features['has_twitter']) and not features['multi_platform']
    
    return features


# ==========================
# Main Collection Pipeline
# ==========================

def collect_social_data(addresses_with_social: pd.DataFrame) -> pd.DataFrame:
    """
    Collect social data for addresses
    
    Expected input DataFrame columns:
    - address: wallet address
    - github_username: (optional) GitHub username
    - twitter_username: (optional) Twitter handle
    """
    
    results = []
    
    print(f"Collecting social data for {len(addresses_with_social)} addresses")
    
    for _, row in tqdm(addresses_with_social.iterrows(), total=len(addresses_with_social), desc="Fetching social profiles"):
        address = row.get('address', row.get('ADDRESS', '')).lower()
        github_user = row.get('github_username', row.get('github', ''))
        twitter_user = row.get('twitter_username', row.get('twitter', ''))
        
        features = {'address': address}
        
        # Fetch GitHub data
        github_profile = None
        if github_user:
            github_profile = get_github_profile(github_user)
            time.sleep(0.5)  # Rate limiting
        
        github_features = extract_github_features(github_profile)
        features.update(github_features)
        
        # Fetch Twitter data
        twitter_profile = None
        if twitter_user:
            twitter_profile = get_twitter_profile_nitter(twitter_user)
            time.sleep(0.5)  # Rate limiting
        
        twitter_features = extract_twitter_features(twitter_profile)
        features.update(twitter_features)
        
        # Cross-platform features
        cross_features = extract_cross_platform_features(github_profile, twitter_profile)
        features.update(cross_features)
        
        results.append(features)
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Collect social profile data")
    parser.add_argument('--input', required=True, help='CSV with addresses and social usernames')
    parser.add_argument('--output', help='Output path (default: data/processed/social_features.csv)')
    
    args = parser.parse_args()
    
    # Load input (handle both CSV and parquet)
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} addresses from {args.input}")
    
    # Check for social columns
    has_github = 'github_username' in df.columns or 'github' in df.columns
    has_twitter = 'twitter_username' in df.columns or 'twitter' in df.columns
    
    if not has_github and not has_twitter:
        print("⚠️  No social username columns found (github_username, twitter_username)")
        print("Expected columns: address, github_username, twitter_username")
        print("\nCreating dummy social features for all addresses...")
        
        # Create empty features
        addresses = df.get('ADDRESS', df.get('address', df.iloc[:, 0])).tolist()
        dummy_features = pd.DataFrame({
            'address': [a.lower() for a in addresses],
            'github_exists': False,
            'twitter_exists': False,
            'multi_platform': False,
            'github_suspicious': True,
            'twitter_suspicious': True,
        })
        
        social_df = dummy_features
    else:
        print(f"Found social columns - GitHub: {has_github}, Twitter: {has_twitter}")
        social_df = collect_social_data(df)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ROOT / 'data' / 'processed' / 'social_features.csv'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    social_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved {len(social_df)} social feature vectors to {output_path}")
    
    if 'github_exists' in social_df.columns:
        print(f"\n📊 Summary:")
        print(f"  GitHub profiles found: {social_df['github_exists'].sum():,}")
        print(f"  Twitter profiles found: {social_df.get('twitter_exists', pd.Series([False])).sum():,}")
        print(f"  Multi-platform users: {social_df.get('multi_platform', pd.Series([False])).sum():,}")
        
        if 'github_suspicious' in social_df.columns:
            print(f"\n🚨 Suspicious signals:")
            print(f"  GitHub suspicious: {social_df['github_suspicious'].sum():,}")
            print(f"  Twitter suspicious: {social_df.get('twitter_suspicious', pd.Series([False])).sum():,}")


if __name__ == '__main__':
    main()
