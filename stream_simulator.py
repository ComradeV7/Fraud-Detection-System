import time
import requests
import random
import pandas as pd
from colorama import Fore, init

init(autoreset=True)

API_URL = "http://localhost:8000/predict"

def generate_stream():
    print(f"{Fore.CYAN}Loading RAW dataset to build the simulation stream...")
    try:
        df_raw = pd.read_csv("data/Base.csv") 
    except FileNotFoundError:
        print(f"{Fore.RED}Error: Could not find 'data/Base.csv'.")
        return

    safe_users_df = df_raw[df_raw['fraud_bool'] == 0].drop(columns=['fraud_bool'])
    fraud_users_df = df_raw[df_raw['fraud_bool'] == 1].drop(columns=['fraud_bool'])

    base_fraudster = fraud_users_df.iloc[0].to_dict()
    safe_users = safe_users_df.head(20).to_dict('records')

    print(f"{Fore.YELLOW}Initializing Multi-Tiered Enterprise Attack Simulation...\n")
    time.sleep(2)

    for i in range(30):
        chance = random.random()
        
        if chance < 0.10:
            # 1. BOT ATTACK (Targets the Gatekeeper)
            payload = random.choice(safe_users).copy()
            payload["transaction_id"] = f"TXN_BOT_{random.randint(1000, 9999)}"
            payload["velocity_6h"] = 99999.0 # Impossible number
            tag = f"{Fore.MAGENTA}[BOT BRUTE FORCE]"
            
        elif chance < 0.25:
            # 2. SOPHISTICATED FRAUD (Targets the ANFIS AI)
            payload = base_fraudster.copy()
            payload["transaction_id"] = f"TXN_FRAUD_{random.randint(1000, 9999)}"
            payload["intended_balcon_amount"] += random.uniform(1, 5)
            tag = f"{Fore.RED}[SOPHISTICATED FRAUD]"
            
        else:
            # 3. NORMAL TRAFFIC
            payload = random.choice(safe_users).copy()
            payload["transaction_id"] = f"TXN_SAFE_{random.randint(1000, 9999)}"
            tag = f"{Fore.GREEN}[NORMAL TRAFFIC]"

        # Fire at the API
        start_time = time.time()
        try:
            response = requests.post(API_URL, json=payload)
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                decision = "🛑 BLOCKED" if result["is_fraud"] else "✅ CLEARED"
                color = Fore.RED if result["is_fraud"] else Fore.GREEN
                
                print(f"{tag} ID: {payload['transaction_id']}")
                print(f"   {color}Decision: {decision} | Source: {result['model_version']} | Latency: {latency:.1f}ms")
                print(f"   {Fore.LIGHTBLACK_EX}XAI: {result['rules_fired'][0]}\n")
            else:
                print(f"{Fore.RED}API Error: {response.text}\n")
                
        except requests.exceptions.ConnectionError:
            print(f"{Fore.RED}Failed to connect to API.\n")
            break

        time.sleep(random.uniform(1.0, 2.5))

if __name__ == "__main__":
    generate_stream()