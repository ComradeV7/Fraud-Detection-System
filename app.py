import streamlit as st
import requests
import uuid
import time
import sqlite3
import pandas as pd
import streamlit.components.v1 as components

# --- Page Config ---
st.set_page_config(page_title="Enterprise Fraud XAI", layout="wide", page_icon="")

st.title("Enterprise Fraud Detection Engine (XAI)")

# =====================================================================
# Top-Level KPI Banner (Reads from SQLite Database)
# =====================================================================
try:
    conn = sqlite3.connect("fraud_logs.db")
    total_fraud_df = pd.read_sql("SELECT SUM(amount) FROM prediction_log WHERE is_fraud = 1", conn)
    avg_latency_df = pd.read_sql("SELECT AVG(latency_ms) FROM prediction_log", conn)
    conn.close()
    
    total_fraud = total_fraud_df.iloc[0,0] if not total_fraud_df.empty and pd.notna(total_fraud_df.iloc[0,0]) else 0
    avg_latency = avg_latency_df.iloc[0,0] if not avg_latency_df.empty and pd.notna(avg_latency_df.iloc[0,0]) else 0
    
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Total Fraud Prevented", value=f"${total_fraud:,.2f}")
    m2.metric(label="Average API Latency", value=f"{avg_latency:.2f} ms")
    m3.metric(label="Active ML Engine", value="v2.0-native-anfis")
    st.markdown("---")
except Exception as e:
    st.warning("Database not found yet. Run a prediction to initialize KPIs.")
    st.markdown("---")

# =====================================================================
# Main Application Tabs
# =====================================================================
tab_manual, tab_live, tab_drift = st.tabs(["Manual Investigation", "Live Threat Monitor", "MLOps & Concept Drift"])

# ---------------------------------------------------------------------
# TAB 1: MANUAL INVESTIGATION (The Form)
# ---------------------------------------------------------------------
with tab_manual:
    st.markdown("#### Enter Transaction Details for AI Analysis")
    
    with st.expander("Personal & Financial Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        income = col1.number_input("Income (Ratio: 0.0 to 1.0)", value=0.9, format="%.2f")
        customer_age = col2.number_input("Customer Age (Years)", value=45)
        employment_status = col3.selectbox("Employment Status (Category)", ["CA", "CB", "CC", "CD", "CE", "CF", "CG"], index=1)
        
        housing_status = col1.selectbox("Housing Status (Category)", ["BA", "BB", "BC", "BD", "BE", "BF", "BG"], index=1)
        prev_address_months_count = col2.number_input("Previous Address Duration (Months)", value=60)
        current_address_months_count = col3.number_input("Current Address Duration (Months)", value=120)
        
        intended_balcon_amount = col1.number_input("Requested Transfer Amount (USD $)", value=25.50)
        proposed_credit_limit = col2.number_input("Proposed Credit Limit (USD $)", value=500.0)
        payment_type = col3.selectbox("Payment Type (Category)", ["AA", "AB", "AC", "AD", "AE"], index=3)
        
        credit_risk_score = col1.number_input("Credit Risk Score (Points)", value=120)
        bank_months_count = col2.number_input("Time with Bank (Months)", value=85)
        has_other_cards = col3.selectbox("Has Other Cards? (0=No, 1=Yes)", [0, 1], index=1)

    with st.expander("Digital Footprint & Device Markers"):
        col4, col5, col6 = st.columns(3)
        name_email_similarity = col4.number_input("Name-Email Similarity (Ratio: 0.0 to 1.0)", value=0.95)
        email_is_free = col5.selectbox("Email is Free Provider? (0=No, 1=Yes)", [0, 1], index=0)
        device_os = col6.selectbox("Device OS", ["windows", "linux", "macintosh", "x11", "other"], index=0)
        
        phone_mobile_valid = col4.selectbox("Mobile Phone Valid? (0=No, 1=Yes)", [0, 1], index=1)
        phone_home_valid = col5.selectbox("Home Phone Valid? (0=No, 1=Yes)", [0, 1], index=1)
        source = col6.selectbox("Request Source", ["INTERNET", "TELEORDER"], index=0)
        
        session_length_in_minutes = col4.number_input("Session Length (Minutes)", value=14.5)
        keep_alive_session = col5.selectbox("Keep Alive Session Active? (0=No, 1=Yes)", [0, 1], index=1)
        device_fraud_count = col6.number_input("Historical Device Fraud Count", value=0)

    with st.expander("Velocity, Geography & Risk Vectors"):
        col7, col8, col9 = st.columns(3)
        velocity_6h = col7.number_input("Transaction Velocity (Last 6 Hours)", value=5.5)
        velocity_24h = col8.number_input("Transaction Velocity (Last 24 Hours)", value=12.0)
        velocity_4w = col9.number_input("Transaction Velocity (Last 4 Weeks)", value=45.0)
        
        zip_count_4w = col7.number_input("Distinct Zip Codes (Last 4 Weeks)", value=1)
        bank_branch_count_8w = col8.number_input("Bank Branches Visited (Last 8 Weeks)", value=0)
        month = col9.number_input("Transaction Month (1-12)", value=7)
        
        device_distinct_emails_8w = col7.number_input("Distinct Emails on Device (Last 8 Weeks)", value=1)
        date_of_birth_distinct_emails_4w = col8.number_input("DOB Distinct Emails (Last 4 Weeks)", value=1)
        foreign_request = col9.selectbox("Foreign IP Request? (0=No, 1=Yes)", [0, 1], index=0)
        
        days_since_request = col7.number_input("Days Since Request (Days)", value=10.0)

    st.markdown("---")

    # --- Submission Logic ---
    if st.button("Run Security Scan", type="primary", use_container_width=True):
        with st.spinner("Analyzing multi-dimensional fuzzy clusters..."):
            payload = {
                "transaction_id": f"TXN_{uuid.uuid4().hex[:8].upper()}",
                "income": income, "name_email_similarity": name_email_similarity,
                "prev_address_months_count": prev_address_months_count, 
                "current_address_months_count": current_address_months_count,
                "customer_age": customer_age, "days_since_request": days_since_request,
                "intended_balcon_amount": intended_balcon_amount, "payment_type": payment_type,
                "zip_count_4w": zip_count_4w, "velocity_6h": velocity_6h, 
                "velocity_24h": velocity_24h, "velocity_4w": velocity_4w,
                "bank_branch_count_8w": bank_branch_count_8w, 
                "date_of_birth_distinct_emails_4w": date_of_birth_distinct_emails_4w,
                "employment_status": employment_status, "credit_risk_score": credit_risk_score,
                "email_is_free": email_is_free, "housing_status": housing_status,
                "phone_home_valid": phone_home_valid, "phone_mobile_valid": phone_mobile_valid,
                "bank_months_count": bank_months_count, "has_other_cards": has_other_cards,
                "proposed_credit_limit": proposed_credit_limit, "foreign_request": foreign_request,
                "source": source, "session_length_in_minutes": session_length_in_minutes,
                "device_os": device_os, "keep_alive_session": keep_alive_session,
                "device_distinct_emails_8w": device_distinct_emails_8w, 
                "device_fraud_count": device_fraud_count, "month": month
            }
            
            try:
                start_req = time.time()
                response = requests.post("http://localhost:8000/predict", json=payload)
                req_time = (time.time() - start_req) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("Security Analysis Results")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Fraud Probability Score", f"{result['fraud_score'] * 100:.2f}%")
                    m2.metric("API Latency", f"{result['latency_ms']} ms")
                    m3.metric("Decision", "REJECTED" if result['is_fraud'] else "APPROVED")
                    
                    if result['is_fraud']:
                        st.error("HIGH RISK ANOMALY DETECTED. TRANSACTION BLOCKED.")
                    else:
                        st.success("TRANSACTION VERIFIED AND CLEARED.")
                    
                    st.info(f"**AI Reasoning:**\n\n{result['rules_fired'][0]}")
                else:
                    st.error(f"API Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend. Is FastAPI running on port 8000?")

# ---------------------------------------------------------------------
# TAB 2: LIVE THREAT MONITOR
# ---------------------------------------------------------------------
with tab_live:
    st.markdown("#### Real-Time Asynchronous Database Stream")
    st.write("Watch this feed populate in real-time as the `stream_simulator.py` attacks the API.")
    
    if st.button("Refresh Live Feed"):
        try:
            conn = sqlite3.connect("fraud_logs.db")
            df_live = pd.read_sql("""
                SELECT timestamp, transaction_id, amount, is_fraud, latency_ms, rules_fired 
                FROM prediction_log 
                ORDER BY timestamp DESC LIMIT 15
            """, conn)
            conn.close()
            
            def highlight_fraud(val):
                color = '#ff4b4b' if val == True else '#00cc66'
                return f'color: {color}; font-weight: bold;'
            
            st.dataframe(
                df_live.style.map(highlight_fraud, subset=['is_fraud']),
                use_container_width=True,
                hide_index=True
            )
        except Exception as e:
            st.warning("No logs found. Start your `stream_simulator.py` script to generate live traffic!")

# ---------------------------------------------------------------------
# TAB 3: MLOPS & CONCEPT DRIFT
# ---------------------------------------------------------------------
with tab_drift:
    st.markdown("#### Statistical Concept Drift Monitor")
    st.write("Evaluates the live SQLite data stream against the PyTorch baseline using Kolmogorov-Smirnov tests.")
    
    if st.button("Load Evidently AI Drift Report"):
        try:
            with open("drift_report.html", "r", encoding="utf-8") as f:
                html_data = f.read()
            components.html(html_data, height=800, scrolling=True)
        except FileNotFoundError:
            st.error("No drift report found. Please run `python drift_monitor.py` in your terminal first.")