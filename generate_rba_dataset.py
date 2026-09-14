# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE RBA (RISK-BASED AUTHENTICATION) DATASET
# Creates realistic authentication logs with attack patterns
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOTAL_SAMPLES = 50000  # Total authentication events
ATTACK_RATIO = 0.15    # 15% attacks (realistic ratio)

# Realistic data pools
COUNTRIES = [
    "US", "GB", "DE", "FR", "CA", "AU", "JP", "IN", "BR", "CN",
    "RU", "KR", "SG", "NL", "SE", "ES", "IT", "MX", "AR", "ZA"
]

DEVICE_TYPES = [
    "Desktop-Windows", "Desktop-Mac", "Desktop-Linux",
    "Mobile-iOS", "Mobile-Android",
    "Tablet-iOS", "Tablet-Android",
    "Server", "API-Client"
]

# Realistic user ID patterns
def generate_user_id(index, is_attack):
    """Generate realistic user IDs"""
    if is_attack:
        # Attackers often use automated patterns
        if random.random() < 0.3:
            return f"user{random.randint(1, 1000):04d}"  # Sequential
        else:
            return f"admin_{random.choice(['test', 'dev', 'prod', '123'])}"  # Common targets
    else:
        # Normal users have more variety
        prefixes = ["user", "employee", "contractor", "guest"]
        return f"{random.choice(prefixes)}{random.randint(1000, 99999)}"

def generate_ip_address(is_attack, country):
    """Generate realistic IP addresses"""
    if is_attack:
        # Known malicious IP ranges (for training purposes)
        if random.random() < 0.4:
            # Tor exit nodes / VPN ranges
            return f"{random.choice([45, 185, 192, 198])}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        else:
            # Random IPs
            return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    else:
        # Corporate IP ranges
        if random.random() < 0.7:
            return f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        else:
            return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

def generate_login_frequency(is_attack):
    """Generate login frequency (logins per day)"""
    if is_attack:
        # Attackers: high frequency (brute force) or very low (reconnaissance)
        if random.random() < 0.7:
            return random.randint(50, 500)  # Brute force
        else:
            return random.randint(1, 5)  # Slow reconnaissance
    else:
        # Normal users: moderate frequency
        return random.randint(1, 20)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generate Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("Generating RBA Authentication Dataset...")
print(f"Total samples: {TOTAL_SAMPLES}")
print(f"Attack ratio: {ATTACK_RATIO * 100}%\n")

data = []

# Determine attack samples
num_attacks = int(TOTAL_SAMPLES * ATTACK_RATIO)
num_normal = TOTAL_SAMPLES - num_attacks

print(f"Normal logins: {num_normal}")
print(f"Attack logins: {num_attacks}\n")

# Generate samples
for i in range(TOTAL_SAMPLES):
    is_attack = 1 if i < num_attacks else 0
    
    # Country selection
    if is_attack:
        # Attacks often come from specific regions
        country = random.choices(
            COUNTRIES,
            weights=[5, 5, 5, 5, 5, 5, 3, 10, 5, 15, 20, 5, 5, 3, 3, 5, 5, 5, 5, 5],
            k=1
        )[0]
    else:
        # Normal traffic is more evenly distributed
        country = random.choice(COUNTRIES)
    
    # Device type
    if is_attack:
        # Attackers prefer automated tools
        device = random.choices(
            DEVICE_TYPES,
            weights=[20, 10, 30, 5, 5, 2, 2, 15, 15],
            k=1
        )[0]
    else:
        # Normal users use various devices
        device = random.choices(
            DEVICE_TYPES,
            weights=[30, 20, 10, 15, 15, 5, 5, 0, 0],
            k=1
        )[0]
    
    # Login success
    if is_attack:
        # Most attacks fail, some succeed
        login_success = 1 if random.random() < 0.05 else 0
    else:
        # Most normal logins succeed
        login_success = 1 if random.random() < 0.95 else 0
    
    # Build record
    record = {
        "user_id": generate_user_id(i, is_attack),
        "ip_address": generate_ip_address(is_attack, country),
        "country": country,
        "device_type": device,
        "login_success": login_success,
        "login_frequency": generate_login_frequency(is_attack),
        "is_attack": is_attack
    }
    
    data.append(record)
    
    if (i + 1) % 10000 == 0:
        print(f"Generated {i + 1}/{TOTAL_SAMPLES} samples...")

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle to mix attacks and normal logins
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Add Realistic Features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\nAdding realistic features...")

# Time-based features
df['hour_of_day'] = [random.randint(0, 23) for _ in range(len(df))]
df['day_of_week'] = [random.randint(0, 6) for _ in range(len(df))]

# Risk score (derived from other features)
def calculate_risk_score(row):
    """Calculate authentication risk score"""
    score = 0
    
    # High login frequency
    if row['login_frequency'] > 50:
        score += 40
    
    # Failed login
    if row['login_success'] == 0:
        score += 20
    
    # Suspicious countries
    if row['country'] in ['RU', 'CN', 'KR']:
        score += 15
    
    # Automated devices
    if 'API' in row['device_type'] or 'Server' in row['device_type']:
        score += 10
    
    # Odd hours (2-6 AM)
    if 2 <= row['hour_of_day'] <= 6:
        score += 15
    
    return min(score, 100)

df['risk_score'] = df.apply(calculate_risk_score, axis=1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Save Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

output_file = "rba.csv"
df.to_csv(output_file, index=False)

print(f"\n{'='*70}")
print("✓ RBA Dataset Generated Successfully")
print(f"{'='*70}\n")

# Statistics
print("Dataset Statistics:")
print(f"  Total records: {len(df)}")
print(f"  Attack samples: {df['is_attack'].sum()} ({df['is_attack'].sum()/len(df)*100:.1f}%)")
print(f"  Normal samples: {(df['is_attack'] == 0).sum()} ({(df['is_attack'] == 0).sum()/len(df)*100:.1f}%)")
print(f"  Successful logins: {df['login_success'].sum()} ({df['login_success'].sum()/len(df)*100:.1f}%)")
print(f"  Failed logins: {(df['login_success'] == 0).sum()} ({(df['login_success'] == 0).sum()/len(df)*100:.1f}%)")
print(f"\n  File saved: {output_file}")
print(f"  File size: {len(df) * df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# Preview
print(f"\nDataset Preview:")
print(df.head(10))

print("\nColumn Information:")
print(df.info())

print("\nAttack Distribution by Country:")
attack_by_country = df[df['is_attack'] == 1]['country'].value_counts().head(10)
print(attack_by_country)

print("\n" + "="*70)
print("✓ Ready to use in training!")
print("="*70)