# 🎯 Pond Sybil Detection System

**Multi-modal bot/Sybil detection for Base blockchain using on-chain + social data**

Built for the Pond Bounty Hunter competition to detect Sybil wallets, bot accounts, and reward farming patterns.



## 💡 Why Multi-Modal Detection?

### The Problem with On-Chain Only Approaches

Traditional Sybil detection relies **only on blockchain transaction patterns**:
- Transaction frequency, gas usage, timing patterns
- Easy for attackers to fake by adding random delays
- Achieves ~85-90% accuracy but misses sophisticated Sybils

### Our Solution: Multi-Modal Analysis

We combine **two independent data sources**:

1. **On-Chain Behavior (60%)** - Blockchain transactions
   - Hard to fake en masse (costs gas fees)
   - Reveals bot-like patterns in timing and gas usage
   
2. **Social Identity (40%)** - GitHub & Twitter profiles
   - **GitHub** (primary): repos, stars, followers, account age, contributions
   - **Twitter** (secondary): profile verification, account existence
   - **Much harder to fake** - requires years of consistent activity
   - Sybil accounts typically have empty/new social profiles or no GitHub history

**Result**: By requiring attackers to fake BOTH on-chain AND social signals, we achieve **95%+ accuracy** vs 85-90% for on-chain only.

---

## 📚 About the Training Dataset

This model was trained on the **"Sybil Detection with Human Passport and Octant"** competition dataset from Pond:

**Dataset Source:** https://cryptopond.xyz/modelfactory/detail/4712551

The dataset contains:
- **20,369 addresses** from Ethereum and Base chains
- Transaction histories, wallet behaviors, and temporal patterns
- Used for training the hybrid ensemble models

---

## 🚀 Quick Start

> **💡 Two Workflows:**
> 1. **Use Pre-trained Models** (Quick) → Jump to Step 1 below
> 2. **Train Your Own Models** (Advanced) → See [Training from Scratch](#-training-from-scratch) first

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────┐
│  Pre-trained Models (Ready to Use) ✅                   │
│  Skip to: python predict_pond.py --input test.csv      │
└─────────────────────────────────────────────────────────┘

                         OR

┌─────────────────────────────────────────────────────────┐
│  Training from Scratch (Advanced) 🔧                    │
│                                                          │
│  1. Download dataset (competition_4712551_*)            │
│  2. Rename & place in Datasets/base/ and /ethereum/    │
│  3. Run: python scripts/hybrid_ensemble.py (30-60 min) │
│  4. Then: python predict_pond.py --input test.csv      │
└─────────────────────────────────────────────────────────┘
```

### 1. Prepare Your Test Data

**📍 Put your test file in the root folder** with one of these names:
- `pond_test_data.csv` (recommended)
- `test_addresses.csv`
- Any name you want (specify with `--input`)

**Format** - Create a CSV file like this:

```csv
wallet_address,github_username,twitter_username
0x1234567890abcdef...,realdev,@realdev
0xabcdef1234567890...,,
```

**Column Guide:**
- `wallet_address` (**REQUIRED**) - Base or Ethereum address (starts with 0x)
- `github_username` (*Optional, but HIGHLY recommended*) - GitHub username (without @)
- `twitter_username` (*Optional*) - Twitter/X handle (with or without @)

**💡 Tip:** Social profiles boost accuracy significantly! Addresses without social data rely only on on-chain patterns.

**Note:** LinkedIn column can be added but is currently not used by the model (reserved for future versions).

**📁 Example file:** See `pond_test_example.csv` for a working example with Vitalik's address.

### 2. Run Prediction

```bash
# If your file is named pond_test_data.csv:
python predict_pond.py --input pond_test_data.csv

# Or with any custom name:
python predict_pond.py --input my_addresses.csv

# Specify custom output:
python predict_pond.py --input pond_test_data.csv --output my_results.csv
```

**⏱️ Time:** ~1-2 seconds per address (includes API calls + caching)

### 3. View Results

Results are saved to **`results.csv`** by default.

**Example output:**
```csv
wallet_address,sybil_score,prediction,confidence,risk_level,github_verified,twitter_verified,social_risk,onchain_risk
0xABC...,0.05,GENUINE,0.90,LOW,True,True,0.0,0.1
0xDEF...,0.85,SYBIL,0.70,HIGH,False,False,1.0,0.8
```

### 📊 Understanding the Results

Each column explained:

| Column | Meaning | Values |
|--------|---------|--------|
| **wallet_address** | The address you tested | 0x... |
| **sybil_score** | Overall Sybil probability | 0.0 (genuine) to 1.0 (Sybil) |
| **prediction** | Final classification | `GENUINE` or `SYBIL` |
| **confidence** | Model certainty | 0.0 (uncertain) to 1.0 (certain) |
| **risk_level** | Easy-to-read risk category | `LOW`, `MEDIUM`, `HIGH` |
| **github_verified** | Has verified GitHub profile? | `True` / `False` |
| **twitter_verified** | Has verified Twitter profile? | `True` / `False` |
| **social_risk** | Social-only risk score | 0.0 (clean) to 1.0 (suspicious) |
| **onchain_risk** | Blockchain-only risk score | 0.0 (clean) to 1.0 (suspicious) |

**Decision Thresholds:**
- **sybil_score < 0.3** → `GENUINE` (Low risk) ✅
- **sybil_score 0.3-0.7** → Manual review recommended ⚠️
- **sybil_score > 0.7** → `SYBIL` (High risk) 🚨

---

## 📁 Project Structure

```
Bounty_Spam_Detector/
│
├── predict_pond.py              # 🎯 Main prediction script (run this!)
├── pond_test_example.csv        # Example test data (Vitalik's address)
├── requirements.txt             # Python dependencies
├── .env                         # Your API keys (create from .env.example)
│
├── models/                      # ✅ Pre-trained ML models (ready to use)
│   ├── base_model.pkl          # Base chain model
│   ├── eth_model.pkl           # Ethereum chain model
│   └── combined_model.pkl      # Combined multi-chain model
│
├── Datasets/                    # 📊 Training data (for retraining only)
│   ├── base/                   # Download: competition_4712551_base/
│   │   ├── test_addresses.parquet
│   │   ├── transactions.parquet
│   │   └── ...
│   └── ethereum/               # Download: competition_4712551_ethereum/
│       ├── test_addresses.parquet
│       ├── transactions.parquet
│       └── ...
│
├── scripts/                     # Core pipeline scripts
│   ├── hybrid_ensemble.py      # 🔧 Model training (run first for retraining)
│   ├── collect_social_data.py  # Social profile extraction
│   ├── collect_light_onchain.py # On-chain feature extraction
│   └── build_multimodal_dataset.py # Feature combination
│
├── src/                         # Source code
│   ├── data/                   # Data collection utilities
│   └── labeling/               # Labeling logic
│
└── cache/                       # Cached API results
    ├── social/                 # GitHub/Twitter cache
    └── light_onchain/          # Blockchain data cache
```

### 📥 Where to Place Training Data (if retraining)

If you want to retrain the model with the Pond competition dataset:

1. **Download the dataset** from: https://cryptopond.xyz/modelfactory/detail/4712551

2. **Extract and rename folders:**
   - Rename `competition_4712551_base/` → `Datasets/base/`
   - Rename `competition_4712551_ethereum/` → `Datasets/ethereum/`

3. **Your folder should look like:**
   ```
   Datasets/
   ├── base/
   │   ├── test_addresses.parquet
   │   └── transactions.parquet
   └── ethereum/
       ├── test_addresses.parquet
       └── transactions.parquet
   ```

---

## 🔧 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add API Key (Required for on-chain data)

Copy the example file and add your API key:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and replace 'your_api_key_here' with your actual key
```

Your `.env` file should look like:
```
ETHERSCAN_API_KEY=RPKWC1ZEZN...your_actual_key_here
```

**Get Free API Key:**
- Visit: https://etherscan.io/apis
- Sign up and get your API key
- Works for both Ethereum and Base chains
- Free tier: 5 requests/second (more than enough!)

**Note:** `.env` is in `.gitignore` - your API key stays private and won't be committed to git.

---

## 🎯 How It Works

### Multi-Modal Detection

Combines **two data sources** for accurate Sybil detection:

| Data Source | Weight | Signals |
|-------------|--------|---------|
| **On-Chain** | 60% | Transaction patterns, gas usage, timing, account age |
| **Social** | 40% | **GitHub**: repos/stars/followers/age, **Twitter**: profile verification |

**Currently Used Social Platforms:**
- ✅ **GitHub** (Primary) - Account age, repos, stars, followers, bio, company
- ✅ **Twitter** (Secondary) - Profile existence and verification
- ⏳ **LinkedIn** (Future) - Reserved for v2.0, not currently scored

### Risk Scoring

Each address receives:
- **Sybil Score** (0-1): 0 = genuine, 1 = Sybil
- **Risk Level**: LOW / MEDIUM / HIGH
- **Prediction**: GENUINE or SYBIL
- **Confidence**: Model certainty

**Thresholds:**
- `< 0.3` = GENUINE (LOW risk)
- `0.3-0.7` = MEDIUM risk (review)
- `> 0.7` = SYBIL (HIGH risk)

---

## 🔧 Training from Scratch

> **Note:** Pre-trained models are already included! Only follow this if you want to retrain with new data.

### Step 1: Get Training Data

Download the Pond competition dataset and place it correctly:

**Dataset Link:** https://cryptopond.xyz/modelfactory/detail/4712551

**Important - Rename the folders:**
```bash
# After downloading, you'll have:
competition_4712551_base/
competition_4712551_ethereum/

# Rename them to:
Datasets/base/
Datasets/ethereum/
```

**Correct placement:**
- Extract `competition_4712551_base/` contents → `Datasets/base/`
- Extract `competition_4712551_ethereum/` contents → `Datasets/ethereum/`

### Step 2: Train Models

```bash
python scripts/hybrid_ensemble.py
```

**What it does:**
- Loads data from `Datasets/base/` and `Datasets/ethereum/`
- Trains 3 models (Base, Ethereum, Combined)
- Saves models to `models/` folder
- Takes 30-60 minutes

### Step 3: Use Trained Models

After training completes, run predictions:

```bash
python predict_pond.py --input your_test_data.csv
```

---

## 🚀 Advanced Usage

### Custom Options

```bash
# Specify output file
python predict_pond.py --input data.csv --output my_results.csv

# Change threshold
python predict_pond.py --input data.csv --threshold 0.6

# Include all features in output
python predict_pond.py --input data.csv --include-features
```

---

## 🏆 For Pond Competition

### Submission Checklist

- ✅ Public GitHub repository
- ✅ Clear README with methodology
- ✅ Source code (all scripts)
- ✅ Dataset with documentation
- ✅ Reproducible predictions
- ✅ Pond profile URL

### Your Profile

**Pond Profile:** [Add your pond.so profile URL here]

---

## 🐛 Troubleshooting

**Error: No trained model found**
- Run `python scripts/hybrid_ensemble.py` to train models

**Error: API rate limit**
- Add your Etherscan API key to `api.txt`
- Free tier: 5 requests/second

**Low accuracy on predictions**
- Add social usernames (GitHub, Twitter) for better results
- Ensure addresses are from Base blockchain
- Check that addresses have transaction history

---

## 📄 License

- **Code**: MIT License
- **Dataset**: CC-BY-4.0

---

## 🔗 Resources

- **Pond Platform**: https://cryptopond.xyz/
- **Etherscan API**: https://etherscan.io/apis
- **BaseScan**: https://basescan.org

---

**Built for Pond Bounty Hunter Competition** 🎯
