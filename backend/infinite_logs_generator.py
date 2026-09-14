"""
Real-Time Synthetic Attack Log Generator with Groq LLM
Generates realistic security logs with randomized attack patterns
Preprocessor will detect CVE and attack types automatically
"""

import random
import json
import time
from datetime import datetime, timedelta
from groq import Groq
import os
import string

# ============================================================================
# INSTALLATION INSTRUCTIONS
# ============================================================================
# Run this command to install required packages:
# pip install groq

# Alternative: Install using requirements.txt
# Create a file named requirements.txt with:
# groq>=0.4.0

# Then run:
# pip install -r requirements.txt

# Set your Groq API key as environment variable:
# export GROQ_API_KEY='your_api_key_here'
# or set it directly in the code below

# ============================================================================
# QUICK START
# ============================================================================
# 1. Install: pip install groq
# 2. Get API key from: https://console.groq.com
# 3. Set API key: export GROQ_API_KEY='your_key_here'
# 4. Run: python log_generator.py

# ============================================================================
# CONFIGURATION
# ============================================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_api_key_here")  # Replace with your Groq API key
OUTPUT_FILE = "security_logs.log"
GENERATION_INTERVAL = 5  # seconds between log generation
LOGS_PER_BATCH = 3  # number of logs to generate per batch

# ============================================================================
# RANDOM DATA GENERATORS
# ============================================================================

def random_ip(ip_type="any"):
    """Generate random IP address"""
    if ip_type == "internal":
        prefixes = ["10.", "172.16.", "192.168."]
        prefix = random.choice(prefixes)
        if prefix == "10.":
            return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        elif prefix == "172.16.":
            return f"172.{random.randint(16,31)}.{random.randint(0,255)}.{random.randint(1,254)}"
        else:
            return f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
    elif ip_type == "external":
        return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    else:
        return random_ip(random.choice(["internal", "external"]))

def random_port():
    """Generate random port number"""
    common_ports = [20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3306, 3389, 5432, 8080, 8443]
    if random.random() > 0.3:
        return random.choice(common_ports)
    return random.randint(1024, 65535)

def random_username():
    """Generate random username"""
    prefixes = ["user", "admin", "svc", "app", "sys", "dev", "prod", "test", "root", "guest"]
    suffixes = ["", str(random.randint(1, 999)), "_" + ''.join(random.choices(string.ascii_lowercase, k=3))]
    return random.choice(prefixes) + random.choice(suffixes)

def random_hostname():
    """Generate random hostname"""
    types = ["web", "db", "app", "api", "auth", "mail", "dns", "file", "backup", "proxy"]
    envs = ["prod", "dev", "test", "staging", "qa"]
    regions = ["us", "eu", "asia", "east", "west", "central"]

    return f"{random.choice(types)}-{random.choice(envs)}-{random.choice(regions)}-{random.randint(1,99)}"

def random_cloud_platform():
    """Generate random cloud platform details"""
    platforms = [
        {
            "provider": "aws",
            "service": random.choice(["EC2", "S3", "Lambda", "RDS", "ECS", "EKS", "CloudFront", "Route53", "IAM", "CloudWatch"]),
            "region": random.choice(["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "ap-northeast-1"]),
            "account_id": ''.join(random.choices(string.digits, k=12)),
            "resource_id": f"i-{random.choice(string.hexdigits[:16]) * 17}"[:17]
        },
        {
            "provider": "gcp",
            "service": random.choice(["Compute Engine", "Cloud Storage", "Cloud Functions", "Cloud SQL", "GKE", "Cloud Run", "BigQuery"]),
            "region": random.choice(["us-central1", "us-east1", "europe-west1", "asia-southeast1", "australia-southeast1"]),
            "project_id": f"project-{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}",
            "resource_id": ''.join(random.choices(string.digits, k=19))
        },
        {
            "provider": "azure",
            "service": random.choice(["Virtual Machines", "Blob Storage", "Azure Functions", "SQL Database", "AKS", "App Service", "Key Vault"]),
            "region": random.choice(["eastus", "westus2", "northeurope", "southeastasia", "australiaeast"]),
            "subscription_id": str(random.randint(10**10, 10**11-1)),
            "resource_id": f"/subscriptions/{random.randint(10**10, 10**11-1)}/resourceGroups/rg-{random.randint(1000,9999)}"
        },
        {
            "provider": "alibaba",
            "service": random.choice(["ECS", "OSS", "Function Compute", "RDS", "Container Service"]),
            "region": random.choice(["cn-hangzhou", "cn-shanghai", "cn-beijing", "ap-southeast-1"]),
            "account_id": ''.join(random.choices(string.digits, k=16)),
            "resource_id": f"ecs-{random.choice(string.ascii_lowercase) * 12}"[:16]
        },
        {
            "provider": "oracle",
            "service": random.choice(["Compute", "Object Storage", "Functions", "Database", "Container Engine"]),
            "region": random.choice(["us-ashburn-1", "us-phoenix-1", "eu-frankfurt-1", "ap-tokyo-1"]),
            "tenancy_id": f"ocid1.tenancy.{''.join(random.choices(string.ascii_lowercase + string.digits, k=20))}",
            "resource_id": f"ocid1.instance.{''.join(random.choices(string.ascii_lowercase + string.digits, k=20))}"
        }
    ]
    return random.choice(platforms)

def random_process_name():
    """Generate random process name"""
    processes = [
        "svchost.exe", "chrome.exe", "firefox.exe", "explorer.exe", "cmd.exe", "powershell.exe",
        "python.exe", "java.exe", "node.exe", "nginx", "apache2", "mysqld", "postgres",
        "dockerd", "kubelet", "containerd", "systemd", "sshd", "bash", "sh"
    ]
    suspicious = [
        "rundll32.exe", "regsvr32.exe", "mshta.exe", "wscript.exe", "cscript.exe",
        "nc.exe", "ncat.exe", "wget", "curl", "certutil.exe", "bitsadmin.exe"
    ]
    return random.choice(processes + suspicious)

def random_file_path():
    """Generate random file path"""
    linux_paths = [
        "/var/log/", "/etc/", "/home/", "/tmp/", "/opt/", "/usr/bin/", "/usr/local/"
    ]
    windows_paths = [
        "C:\\Windows\\System32\\", "C:\\Users\\", "C:\\Temp\\", "C:\\Program Files\\",
        "C:\\ProgramData\\", "C:\\Windows\\Temp\\"
    ]

    if random.random() > 0.5:
        base = random.choice(linux_paths)
        return base + ''.join(random.choices(string.ascii_lowercase, k=8)) + random.choice([".log", ".conf", ".sh", ".py"])
    else:
        base = random.choice(windows_paths)
        return base + ''.join(random.choices(string.ascii_lowercase, k=8)) + random.choice([".exe", ".dll", ".bat", ".ps1"])

def random_url():
    """Generate random URL"""
    domains = ["example.com", "test.net", "sample.org", "demo.io", "suspicious-site.com"]
    paths = ["login", "admin", "api", "upload", "download", "config", "backup"]
    return f"http{'s' if random.random() > 0.3 else ''}://{random.choice(domains)}/{random.choice(paths)}?id={random.randint(1,9999)}"

def random_user_agent():
    """Generate random user agent"""
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "python-requests/2.28.0",
        "curl/7.68.0",
        "Suspicious-Scanner/1.0"
    ]
    return random.choice(agents)

# ============================================================================
# GROQ CLIENT INITIALIZATION
# ============================================================================

try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"⚠️  Error initializing Groq client: {e}")
    print("Please set GROQ_API_KEY environment variable or update the code")
    exit(1)

# ============================================================================
# ATTACK SCENARIO GENERATION WITH GROQ
# ============================================================================

def clean_text(text):
    """Remove problematic Unicode characters"""
    if not text:
        return ""
    # Remove non-ASCII characters that cause encoding issues
    return text.encode('ascii', 'ignore').decode('ascii')

def generate_random_attack_scenario():
    """Generate completely random attack scenario using Groq LLM"""

    prompt = """Generate a random cybersecurity attack scenario with completely random characteristics.

Create a unique attack with:
1. Random attack behavior/pattern (don't name specific attack types, just describe the behavior)
2. Random suspicious activities
3. Random network patterns
4. Random system behaviors
5. Random indicators of compromise

Return ONLY valid JSON in this exact format (use only ASCII characters):
{
    "behavior_description": "detailed description of what the attack does",
    "network_pattern": "unusual network behavior observed",
    "system_indicators": "suspicious system-level activities",
    "data_pattern": "unusual data access or transfer patterns",
    "severity_score": 7,
    "risk_level": "HIGH",
    "affected_component": "component being targeted",
    "attack_vector": "how the attack is being executed",
    "payload_signature": "unique signature or pattern of the attack",
    "anomaly_score": 0.85
}

Make it completely random and unique each time. Don't use common attack names. Use simple ASCII characters only."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a cybersecurity AI that generates random attack scenarios. Always return valid JSON only, no extra text. Make each scenario unique and random. Use only ASCII characters."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="openai/gpt-oss-20b",  # Using OpenAI model via Groq
            temperature=1.2,  # Higher temperature for more randomness
            max_tokens=600
        )

        response_text = chat_completion.choices[0].message.content.strip()

        # Extract JSON if wrapped in markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        # Clean the response text
        response_text = clean_text(response_text)

        attack_data = json.loads(response_text)

        # Clean all text fields
        for key, value in attack_data.items():
            if isinstance(value, str):
                attack_data[key] = clean_text(value)

        return attack_data

    except Exception as e:
        print(f"⚠️  Groq API error: {e}")
        # Fallback to generic random data
        return {
            "behavior_description": "Unusual sequence of system calls detected",
            "network_pattern": f"Multiple connections to port {random_port()}",
            "system_indicators": f"Process {random_process_name()} spawned unusual child processes",
            "data_pattern": f"Large data transfer of {random.randint(1, 500)}MB detected",
            "severity_score": random.randint(5, 10),
            "risk_level": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
            "affected_component": random_hostname(),
            "attack_vector": random.choice(["network", "application", "system", "user"]),
            "payload_signature": ''.join(random.choices(string.hexdigits.lower(), k=32)),
            "anomaly_score": round(random.uniform(0.6, 0.99), 2)
        }

# ============================================================================
# LOG GENERATION
# ============================================================================

def generate_benign_log(timestamp):
    """Generate a benign/normal log entry"""

    cloud = random_cloud_platform()

    log_entry = {
        "timestamp": timestamp,
        "level": random.choice(["INFO", "DEBUG"]),  
        "source_ip": random_ip("internal"),
        "destination_ip": random_ip(),
        "source_port": random_port(),
        "destination_port": random_port(),
        "user": random_username(),
        "hostname": random_hostname(),
        "process": random_process_name(),
        "protocol": random.choice(["TCP", "UDP", "HTTP", "HTTPS", "SSH", "DNS"]),
        "action": random.choice(["ALLOW", "PERMIT", "SUCCESS", "ACCEPT"]),
        "bytes_sent": random.randint(100, 10000),
        "bytes_received": random.randint(100, 10000),
        "duration_ms": random.randint(10, 500),
        "cloud_provider": cloud["provider"],
        "cloud_service": cloud["service"],
        "cloud_region": cloud["region"],
        "resource_id": cloud["resource_id"],
        "event_category": random.choice(["authentication", "file_access", "network_connection", "process_execution"]),
        "event_outcome": "success",
        "label": "BENIGN"
    }

    # Add provider-specific fields
    if cloud["provider"] == "aws":
        log_entry["aws_account_id"] = cloud["account_id"]
    elif cloud["provider"] == "gcp":
        log_entry["gcp_project_id"] = cloud["project_id"]
    elif cloud["provider"] == "azure":
        log_entry["azure_subscription_id"] = cloud["subscription_id"]

    return log_entry

def generate_malicious_log(timestamp):
    """Generate a malicious log entry with Groq-generated random attack"""

    # Get random attack scenario from Groq
    attack_info = generate_random_attack_scenario()

    cloud = random_cloud_platform()

    log_entry = {
        "timestamp": timestamp,
        "level": random.choice(["ALERT", "CRITICAL", "WARNING"]),
        "source_ip": random_ip("external"),
        "destination_ip": random_ip("internal"),
        "source_port": random_port(),
        "destination_port": random_port(),
        "user": random_username(),
        "hostname": random_hostname(),
        "process": random_process_name(),
        "protocol": random.choice(["TCP", "UDP", "HTTP", "HTTPS", "SSH", "FTP", "SMB"]),
        "action": random.choice(["BLOCK", "DENY", "DROP", "ALERT", "ALLOW"]),
        "bytes_sent": random.randint(1000, 500000),
        "bytes_received": random.randint(1000, 500000),
        "duration_ms": random.randint(100, 5000),
        "cloud_provider": cloud["provider"],
        "cloud_service": cloud["service"],
        "cloud_region": cloud["region"],
        "resource_id": cloud["resource_id"],

        # Attack-specific fields (for preprocessor to analyze)
        "behavior_description": attack_info.get("behavior_description", ""),
        "network_pattern": attack_info.get("network_pattern", ""),
        "system_indicators": attack_info.get("system_indicators", ""),
        "data_pattern": attack_info.get("data_pattern", ""),
        "severity_score": attack_info.get("severity_score", random.randint(5, 10)),
        "risk_level": attack_info.get("risk_level", "HIGH"),
        "affected_component": attack_info.get("affected_component", random_hostname()),
        "attack_vector": attack_info.get("attack_vector", "network"),
        "payload_signature": attack_info.get("payload_signature", ""),
        "anomaly_score": attack_info.get("anomaly_score", round(random.uniform(0.7, 0.99), 2)),

        # Additional random indicators
        "failed_login_attempts": random.randint(1, 100) if random.random() > 0.5 else None,
        "suspicious_file_path": random_file_path() if random.random() > 0.5 else None,
        "suspicious_url": random_url() if random.random() > 0.5 else None,
        "user_agent": random_user_agent() if random.random() > 0.5 else None,
        "command_line": f"{random_process_name()} {' '.join(random.choices(['-c', '--config', '-e', '--exec', '-f'], k=2))}" if random.random() > 0.5 else None,
        "registry_key_modified": random.random() > 0.7,
        "unusual_time_access": random.random() > 0.8,

        "event_category": random.choice(["intrusion_detection", "malware", "policy_violation", "data_exfiltration", "privilege_escalation"]),
        "event_outcome": random.choice(["failure", "blocked", "detected", "quarantined"]),
        "label": "MALICIOUS"
    }

    # Add provider-specific fields
    if cloud["provider"] == "aws":
        log_entry["aws_account_id"] = cloud["account_id"]
    elif cloud["provider"] == "gcp":
        log_entry["gcp_project_id"] = cloud["project_id"]
    elif cloud["provider"] == "azure":
        log_entry["azure_subscription_id"] = cloud["subscription_id"]

    return log_entry

# ============================================================================
# LOG FORMATTING (UDM-like format)
# ============================================================================

def format_log_line(log_entry):
    """Format log entry in UDM-compatible format"""

    # Clean all string values
    def clean_value(val):
        if isinstance(val, str):
            return clean_text(val).replace('"', "'")  # Replace quotes to prevent format issues
        return val

    # Build base log line
    log_parts = [
        f"timestamp=\"{clean_value(log_entry['timestamp'])}\"",
        f"level=\"{clean_value(log_entry['level'])}\"",
        f"source.ip=\"{clean_value(log_entry['source_ip'])}\"",
        f"destination.ip=\"{clean_value(log_entry['destination_ip'])}\"",
        f"source.port={log_entry['source_port']}",
        f"destination.port={log_entry['destination_port']}",
        f"user.name=\"{clean_value(log_entry['user'])}\"",
        f"host.name=\"{clean_value(log_entry['hostname'])}\"",
        f"process.name=\"{clean_value(log_entry['process'])}\"",
        f"network.protocol=\"{clean_value(log_entry['protocol'])}\"",
        f"event.action=\"{clean_value(log_entry['action'])}\"",
        f"network.bytes_sent={log_entry['bytes_sent']}",
        f"network.bytes_received={log_entry['bytes_received']}",
        f"event.duration_ms={log_entry['duration_ms']}",
        f"cloud.provider=\"{clean_value(log_entry['cloud_provider'])}\"",
        f"cloud.service=\"{clean_value(log_entry['cloud_service'])}\"",
        f"cloud.region=\"{clean_value(log_entry['cloud_region'])}\"",
        f"cloud.resource_id=\"{clean_value(log_entry['resource_id'])}\"",
        f"event.category=\"{clean_value(log_entry['event_category'])}\"",
        f"event.outcome=\"{clean_value(log_entry['event_outcome'])}\""
    ]

    # Add cloud-specific identifiers
    if "aws_account_id" in log_entry:
        log_parts.append(f"cloud.account.id=\"{clean_value(log_entry['aws_account_id'])}\"")
    elif "gcp_project_id" in log_entry:
        log_parts.append(f"cloud.project.id=\"{clean_value(log_entry['gcp_project_id'])}\"")
    elif "azure_subscription_id" in log_entry:
        log_parts.append(f"cloud.subscription.id=\"{clean_value(log_entry['azure_subscription_id'])}\"")

    # Add malicious-specific fields
    if log_entry['label'] == "MALICIOUS":
        if log_entry.get('behavior_description'):
            desc = clean_value(log_entry['behavior_description'])[:150]
            log_parts.append(f"threat.behavior=\"{desc}\"")
        if log_entry.get('network_pattern'):
            log_parts.append(f"threat.network_pattern=\"{clean_value(log_entry['network_pattern'])}\"")
        if log_entry.get('system_indicators'):
            indicators = clean_value(log_entry['system_indicators'])[:150]
            log_parts.append(f"threat.system_indicators=\"{indicators}\"")
        if log_entry.get('data_pattern'):
            log_parts.append(f"threat.data_pattern=\"{clean_value(log_entry['data_pattern'])}\"")
        if log_entry.get('severity_score'):
            log_parts.append(f"threat.severity={log_entry['severity_score']}")
        if log_entry.get('risk_level'):
            log_parts.append(f"threat.risk_level=\"{clean_value(log_entry['risk_level'])}\"")
        if log_entry.get('payload_signature'):
            log_parts.append(f"threat.signature=\"{clean_value(log_entry['payload_signature'])}\"")
        if log_entry.get('anomaly_score'):
            log_parts.append(f"threat.anomaly_score={log_entry['anomaly_score']}")
        if log_entry.get('failed_login_attempts'):
            log_parts.append(f"auth.failed_attempts={log_entry['failed_login_attempts']}")
        if log_entry.get('suspicious_file_path'):
            log_parts.append(f"file.path=\"{clean_value(log_entry['suspicious_file_path'])}\"")
        if log_entry.get('suspicious_url'):
            log_parts.append(f"url.full=\"{clean_value(log_entry['suspicious_url'])}\"")
        if log_entry.get('user_agent'):
            log_parts.append(f"user_agent.original=\"{clean_value(log_entry['user_agent'])}\"")
        if log_entry.get('command_line'):
            log_parts.append(f"process.command_line=\"{clean_value(log_entry['command_line'])}\"")
        if log_entry.get('registry_key_modified'):
            log_parts.append(f"registry.modified=true")
        if log_entry.get('unusual_time_access'):
            log_parts.append(f"threat.unusual_timing=true")

    log_parts.append(f"labels.detection=\"{clean_value(log_entry['label'])}\"")

    return " ".join(log_parts)

# ============================================================================
# MAIN GENERATOR
# ============================================================================

def generate_and_write_logs(malicious_ratio=0.4):
    """Generate logs continuously and write to file"""

    print(f"🚀 Starting Real-Time Random Attack Log Generator")
    print(f"📁 Output file: {OUTPUT_FILE}")
    print(f"⏱️  Generating {LOGS_PER_BATCH} logs every {GENERATION_INTERVAL} seconds")
    print(f"⚠️  Malicious ratio: {malicious_ratio * 100}%")
    print(f"🔀 All attack types, IPs, users, and cloud platforms are RANDOMIZED")
    print(f"🔍 Preprocessor will detect CVE and attack patterns from log behavior")
    print(f"🛑 Press Ctrl+C to stop\n")

    log_count = 0
    malicious_count = 0
    benign_count = 0
    start_time = datetime.now()

    try:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:  # Add UTF-8 encoding
            # Write header
            f.write(f"# Security Log Generator - Started at {start_time.isoformat()}\n")
            f.write("# UDM-Compatible Format: Preprocessor will detect attack types and CVEs from behavior patterns\n")
            f.write("# All IPs, users, hostnames, and cloud platforms are randomized\n\n")

            while True:
                current_time = datetime.now()

                for _ in range(LOGS_PER_BATCH):
                    # Decide if this log should be malicious
                    is_malicious = random.random() < malicious_ratio

                    # Generate log entry
                    if is_malicious:
                        log_entry = generate_malicious_log(current_time.isoformat())
                        malicious_count += 1
                        print(f"🔴 MALICIOUS [{malicious_count}]: {log_entry['cloud_provider'].upper()} | {log_entry['cloud_service']} | Risk: {log_entry.get('risk_level', 'N/A')}")
                    else:
                        log_entry = generate_benign_log(current_time.isoformat())
                        benign_count += 1
                        print(f"🟢 BENIGN [{benign_count}]: {log_entry['cloud_provider'].upper()} | {log_entry['cloud_service']}")

                    # Format and write to file
                    log_line = format_log_line(log_entry)
                    f.write(log_line + "\n")
                    f.flush()  # Ensure it's written immediately

                    log_count += 1

                    # Small delay between logs in batch
                    time.sleep(0.5)

                print(f"📊 Total: {log_count} | Malicious: {malicious_count} | Benign: {benign_count}\n")

                # Wait before next batch
                time.sleep(GENERATION_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n🛑 Generator stopped by user")
        print(f"📊 Final Statistics:")
        print(f"   Total logs: {log_count}")
        print(f"   Malicious: {malicious_count} ({malicious_count/log_count*100:.1f}%)")
        print(f"   Benign: {benign_count} ({benign_count/log_count*100:.1f}%)")
        print(f"⏱️  Runtime: {datetime.now() - start_time}")
        print(f"📁 Logs saved to: {OUTPUT_FILE}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🔐 REAL-TIME SECURITY LOG GENERATOR")
    print("=" * 70)
    print(f"✅ API Key: Configured (gsk_...)")
    print(f"🤖 Model: openai/gpt-oss-20b")
    print(f"📁 Output: {OUTPUT_FILE}")
    print("=" * 70)
    print()

    # Start generating logs
    generate_and_write_logs(malicious_ratio=0.4)