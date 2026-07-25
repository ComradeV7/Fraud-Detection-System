"""
Real-Time Fraud Detection Dashboard - PostgreSQL Streaming Edition
Displays live fraud detection data from the streaming pipeline
"""
import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# --- Page Config ---
st.set_page_config(
    page_title="Real-Time Fraud Detection", 
    layout="wide", 
    page_icon="🔍"
)

# --- Database Connection ---
@st.cache_resource
def get_db_connection():
    """Create PostgreSQL connection with autocommit"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='fraud_detection',
            user='fraud_user',
            password='fraud_password'
        )
        conn.autocommit = True  # Prevent transaction lock issues
        return conn
    except Exception as e:
        st.error(f"Failed to connect to PostgreSQL: {e}")
        return None

# --- Data Fetching Functions ---
def get_realtime_metrics(conn):
    """Get real-time metrics from last hour"""
    query = """
        SELECT 
            COUNT(*) as total_transactions,
            COUNT(*) FILTER (WHERE fd.is_fraud = true) as fraud_count,
            CASE 
                WHEN COUNT(*) > 0 THEN ROUND(100.0 * COUNT(*) FILTER (WHERE fd.is_fraud = true) / COUNT(*), 2)
                ELSE 0
            END as fraud_rate,
            ROUND(AVG(fd.fraud_score), 4) as avg_fraud_score,
            ROUND(AVG(fd.processing_time_ms), 2) as avg_latency_ms,
            ROUND(AVG(fd.confidence_level), 4) as avg_confidence,
            COALESCE(SUM(t.amount) FILTER (WHERE fd.is_fraud = true), 0) as total_fraud_blocked,
            MAX(t.timestamp) as last_transaction_time
        FROM fraud_decisions fd
        JOIN transactions t ON fd.transaction_id = t.transaction_id
        WHERE fd.created_at >= NOW() - INTERVAL '1 hour'
    """
    try:
        df = pd.read_sql(query, conn)
        return df.iloc[0] if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")
        conn.rollback()
        return None

def get_live_feed(conn, limit=20):
    """Get recent transactions with fraud decisions"""
    query = """
        SELECT 
            t.transaction_id,
            t.timestamp,
            t.amount,
            t.user_id,
            t.merchant_category,
            fd.fraud_score,
            fd.is_fraud,
            fd.confidence_level,
            fd.features_used,
            fd.rules_triggered,
            fd.processing_time_ms,
            fd.model_version
        FROM transactions t
        JOIN fraud_decisions fd ON t.transaction_id = fd.transaction_id
        ORDER BY t.timestamp DESC
        LIMIT %s
    """
    try:
        df = pd.read_sql(query, conn, params=(limit,))
        return df
    except Exception as e:
        st.error(f"Error fetching live feed: {e}")
        conn.rollback()
        return pd.DataFrame()

def get_fraud_rate_over_time(conn, hours=24):
    """Get fraud rate by hour"""
    query = """
        SELECT 
            DATE_TRUNC('hour', t.timestamp) as hour,
            COUNT(*) as total_txns,
            COUNT(*) FILTER (WHERE fd.is_fraud = true) as fraud_txns,
            CASE 
                WHEN COUNT(*) > 0 THEN ROUND(100.0 * COUNT(*) FILTER (WHERE fd.is_fraud = true) / COUNT(*), 2)
                ELSE 0
            END as fraud_rate
        FROM transactions t
        JOIN fraud_decisions fd ON t.transaction_id = fd.transaction_id
        WHERE t.timestamp >= NOW() - INTERVAL '%s hours'
        GROUP BY DATE_TRUNC('hour', t.timestamp)
        ORDER BY hour DESC
    """
    try:
        df = pd.read_sql(query, conn, params=(hours,))
        return df
    except Exception as e:
        st.error(f"Error fetching fraud rate over time: {e}")
        conn.rollback()
        return pd.DataFrame()

def get_risk_distribution(conn):
    """Get transaction distribution by risk level"""
    query = """
        SELECT 
            CASE 
                WHEN fraud_score >= 0.9 THEN 'Very High Risk (0.9+)'
                WHEN fraud_score >= 0.7 THEN 'High Risk (0.7-0.9)'
                WHEN fraud_score >= 0.5 THEN 'Medium Risk (0.5-0.7)'
                WHEN fraud_score >= 0.3 THEN 'Low Risk (0.3-0.5)'
                ELSE 'Very Low Risk (<0.3)'
            END as risk_category,
            COUNT(*) as count,
            ROUND(AVG(fraud_score), 4) as avg_score,
            ROUND(AVG(t.amount), 2) as avg_amount
        FROM fraud_decisions fd
        JOIN transactions t ON fd.transaction_id = t.transaction_id
        GROUP BY risk_category
        ORDER BY MIN(fraud_score) DESC
    """
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching risk distribution: {e}")
        conn.rollback()
        return pd.DataFrame()

def get_top_fraud_features(conn, limit=10):
    """Get most common fraud-driving features"""
    query = """
        SELECT 
            features_used->>0 as primary_feature,
            features_used->>1 as secondary_feature,
            COUNT(*) as occurrence_count,
            ROUND(AVG(fraud_score), 4) as avg_fraud_score,
            ROUND(AVG(t.amount), 2) as avg_amount
        FROM fraud_decisions fd
        JOIN transactions t ON fd.transaction_id = t.transaction_id
        WHERE is_fraud = true
        GROUP BY features_used->>0, features_used->>1
        ORDER BY occurrence_count DESC
        LIMIT %s
    """
    try:
        df = pd.read_sql(query, conn, params=(limit,))
        return df
    except Exception as e:
        st.error(f"Error fetching top fraud features: {e}")
        conn.rollback()
        return pd.DataFrame()

def get_score_distribution(conn):
    """Get fraud score distribution"""
    query = """
        SELECT fraud_score, COUNT(*) as count
        FROM fraud_decisions
        GROUP BY fraud_score
        ORDER BY fraud_score
    """
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching score distribution: {e}")
        conn.rollback()
        return pd.DataFrame()

def get_high_risk_transactions(conn, limit=10):
    """Get highest risk transactions"""
    query = """
        SELECT 
            t.transaction_id,
            t.amount,
            t.timestamp,
            fd.fraud_score,
            fd.is_fraud,
            fd.features_used->>0 as top_feature,
            fd.rules_triggered[1] as explanation
        FROM transactions t
        JOIN fraud_decisions fd ON t.transaction_id = t.transaction_id
        WHERE fd.is_fraud = true
        ORDER BY fd.fraud_score DESC, t.amount DESC
        LIMIT %s
    """
    df = pd.read_sql(query, conn, params=(limit,))
    return df

# =====================================================================
# MAIN DASHBOARD
# =====================================================================

st.title("Real-Time Fraud Detection Dashboard")
st.markdown("**Live streaming data from PostgreSQL** | Auto-refresh every 5 seconds")

# Get database connection
conn = get_db_connection()

if conn is None:
    st.error("Cannot connect to database. Please check PostgreSQL is running.")
    st.stop()

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
if auto_refresh:
    st_autorefresh = st.empty()
    import time
    time.sleep(5)
    st.rerun()

# Refresh button
if st.sidebar.button("Refresh Now", use_container_width=True):
    st.rerun()

# Time range selector
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 1 Hour", "Last 24 Hours", "Last 7 Days", "All Time"],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard Mode")
st.sidebar.info("**Real-Time Mode:** Connected to PostgreSQL streaming pipeline")

# =====================================================================
# KPI METRICS
# =====================================================================
st.markdown("### Real-Time Metrics")

try:
    metrics = get_realtime_metrics(conn)
    
    if metrics is not None and not metrics.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Total Transactions",
            f"{int(metrics['total_transactions']):,}",
            help="Transactions processed in last hour"
        )
        
        col2.metric(
            "Fraud Rate",
            f"{float(metrics['fraud_rate']):.1f}%",
            delta=f"{int(metrics['fraud_count'])} fraud cases",
            help="Percentage of transactions flagged as fraud"
        )
        
        col3.metric(
            "Avg Latency",
            f"{float(metrics['avg_latency_ms']):.2f} ms" if pd.notna(metrics['avg_latency_ms']) else "N/A",
            help="Average processing time per transaction"
        )
        
        col4.metric(
            "Fraud Blocked",
            f"${float(metrics['total_fraud_blocked']):,.2f}" if pd.notna(metrics['total_fraud_blocked']) else "$0.00",
            help="Total fraud amount prevented"
        )
        
        # Additional metrics row
        col5, col6, col7, col8 = st.columns(4)
        
        col5.metric(
            "Avg Fraud Score",
            f"{float(metrics['avg_fraud_score']):.4f}" if pd.notna(metrics['avg_fraud_score']) else "N/A",
            help="Average fraud probability score"
        )
        
        col6.metric(
            "Avg Confidence",
            f"{float(metrics['avg_confidence']):.2%}" if pd.notna(metrics['avg_confidence']) else "N/A",
            help="Average model confidence level"
        )
        
        col7.metric(
            "Last Transaction",
            metrics['last_transaction_time'].strftime("%H:%M:%S") if pd.notna(metrics['last_transaction_time']) else "N/A",
            help="Most recent transaction timestamp"
        )
        
        col8.metric(
            "Model Version",
            "v2.0-xgboost",
            help="Active ML model version"
        )
    else:
        st.warning("No data in last hour. Run the streaming pipeline to see live data!")

except Exception as e:
    st.error(f"Error loading metrics: {e}")

st.markdown("---")

# =====================================================================
# MAIN TABS
# =====================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Live Transaction Feed", 
    "Risk Analysis", 
    "Explainability", 
    "Performance"
])

# ---------------------------------------------------------------------
# TAB 1: LIVE TRANSACTION FEED
# ---------------------------------------------------------------------
with tab1:
    st.markdown("### Live Transaction Stream")
    
    try:
        live_feed = get_live_feed(conn, limit=30)
        
        if not live_feed.empty:
            # Display count
            st.info(f"Showing last **{len(live_feed)}** transactions")
            
            # Format dataframe for display
            display_df = live_feed.copy()
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['fraud_score'] = display_df['fraud_score'].apply(lambda x: f"{x:.4f}")
            display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
            display_df['confidence_level'] = display_df['confidence_level'].apply(lambda x: f"{x:.2%}")
            display_df['is_fraud'] = display_df['is_fraud'].apply(lambda x: "FRAUD" if x else "LEGIT")
            
            # Color coding function
            def highlight_fraud(row):
                if "FRAUD" in str(row['is_fraud']):
                    return ['background-color: #ff4b4b; color: white'] * len(row)
                else:
                    return ['background-color: #00cc66; color: white'] * len(row)
            
            # Display styled dataframe
            st.dataframe(
                display_df[['timestamp', 'transaction_id', 'amount', 'fraud_score', 'is_fraud', 'confidence_level', 'processing_time_ms']]
                .style.apply(highlight_fraud, axis=1),
                use_container_width=True,
                hide_index=True,
                height=600
            )
            
            # Expandable details
            st.markdown("### Transaction Details")
            selected_txn = st.selectbox(
                "Select transaction to view details:",
                live_feed['transaction_id'].tolist()
            )
            
            if selected_txn:
                txn_details = live_feed[live_feed['transaction_id'] == selected_txn].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Transaction Info")
                    st.write(f"**ID:** {txn_details['transaction_id']}")
                    st.write(f"**Amount:** ${txn_details['amount']:,.2f}")
                    st.write(f"**User:** {txn_details['user_id']}")
                    st.write(f"**Category:** {txn_details['merchant_category']}")
                    st.write(f"**Timestamp:** {txn_details['timestamp']}")
                
                with col2:
                    st.markdown("#### Fraud Analysis")
                    st.write(f"**Fraud Score:** {txn_details['fraud_score']:.4f}")
                    st.write(f"**Decision:** {'FRAUD' if txn_details['is_fraud'] else 'LEGITIMATE'}")
                    st.write(f"**Confidence:** {txn_details['confidence_level']:.2%}")
                    st.write(f"**Latency:** {txn_details['processing_time_ms']:.2f} ms")
                    st.write(f"**Model:** {txn_details['model_version']}")
                
                # SHAP Explanation
                st.markdown("#### SHAP Explanation")
                
                # Handle features_used (can be list, string, array, or None)
                try:
                    features_value = txn_details.get('features_used')
                    
                    # Convert to scalar if it's a numpy/pandas array
                    if hasattr(features_value, '__iter__') and not isinstance(features_value, (str, list)):
                        features_value = features_value.item() if hasattr(features_value, 'item') else None
                    
                    if features_value is not None and str(features_value) != 'nan':
                        if isinstance(features_value, str):
                            try:
                                features = json.loads(features_value)
                            except:
                                features = []
                        elif isinstance(features_value, (list, tuple)):
                            features = list(features_value)
                        else:
                            features = []
                        
                        if features and len(features) > 0:
                            st.write("**Top Contributing Features:**")
                            for i, feature in enumerate(features[:3], 1):
                                st.write(f"{i}. `{feature}`")
                        else:
                            st.write("No feature information available")
                    else:
                        st.write("No feature information available")
                except Exception as e:
                    st.write(f"Feature loading error: {str(e)}")
                
                # Handle rules_triggered (can be list, string, array, or None)
                try:
                    rules_value = txn_details.get('rules_triggered')
                    
                    # Convert to scalar if it's a numpy/pandas array
                    if hasattr(rules_value, '__iter__') and not isinstance(rules_value, (str, list)):
                        rules_value = rules_value.item() if hasattr(rules_value, 'item') else None
                    
                    if rules_value is not None and str(rules_value) != 'nan':
                        if isinstance(rules_value, str):
                            # PostgreSQL array format: {item1,item2}
                            if rules_value.startswith('{') and rules_value.endswith('}'):
                                rules = [r.strip().strip('"') for r in rules_value[1:-1].split(',')]
                            else:
                                rules = [rules_value]
                        elif isinstance(rules_value, (list, tuple)):
                            rules = list(rules_value)
                        else:
                            rules = []
                        
                        if rules and len(rules) > 0 and rules[0]:
                            st.info(f"**Explanation:** {rules[0]}")
                        else:
                            st.info("**Explanation:** No explanation available")
                    else:
                        st.info("**Explanation:** No explanation available")
                except Exception as e:
                    st.info(f"**Explanation:** Error - {str(e)}")
        
        else:
            st.warning("No transactions found. Start the streaming pipeline to see live data!")
    
    except Exception as e:
        st.error(f"Error loading live feed: {e}")
        import traceback
        st.code(traceback.format_exc())

# ---------------------------------------------------------------------
# TAB 2: RISK ANALYSIS
# ---------------------------------------------------------------------
with tab2:
    st.markdown("### Risk Distribution & Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Risk Level Distribution")
        try:
            risk_dist = get_risk_distribution(conn)
            
            if not risk_dist.empty:
                # Pie chart
                fig = px.pie(
                    risk_dist,
                    values='count',
                    names='risk_category',
                    title='Transaction Distribution by Risk Level',
                    color='risk_category',
                    color_discrete_map={
                        'Very High Risk (0.9+)': '#ff0000',
                        'High Risk (0.7-0.9)': '#ff6600',
                        'Medium Risk (0.5-0.7)': '#ffcc00',
                        'Low Risk (0.3-0.5)': '#66cc00',
                        'Very Low Risk (<0.3)': '#00cc66'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.dataframe(risk_dist, use_container_width=True, hide_index=True)
            else:
                st.warning("No risk distribution data available")
        
        except Exception as e:
            st.error(f"Error loading risk distribution: {e}")
    
    with col2:
        st.markdown("#### Fraud Rate Over Time")
        try:
            fraud_rate_time = get_fraud_rate_over_time(conn, hours=24)
            
            if not fraud_rate_time.empty:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=fraud_rate_time['hour'],
                    y=fraud_rate_time['fraud_rate'],
                    mode='lines+markers',
                    name='Fraud Rate (%)',
                    line=dict(color='#ff4b4b', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title='Fraud Rate by Hour (Last 24h)',
                    xaxis_title='Hour',
                    yaxis_title='Fraud Rate (%)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No temporal data available")
        
        except Exception as e:
            st.error(f"Error loading fraud rate over time: {e}")
    
    # High-risk transactions table
    st.markdown("#### Highest Risk Transactions")
    try:
        high_risk = get_high_risk_transactions(conn, limit=15)
        
        if not high_risk.empty:
            high_risk['amount'] = high_risk['amount'].apply(lambda x: f"${x:,.2f}")
            high_risk['fraud_score'] = high_risk['fraud_score'].apply(lambda x: f"{x:.4f}")
            high_risk['timestamp'] = pd.to_datetime(high_risk['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(
                high_risk,
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.warning("No high-risk transactions found")
    
    except Exception as e:
        st.error(f"Error loading high-risk transactions: {e}")

# ---------------------------------------------------------------------
# TAB 3: EXPLAINABILITY (SHAP)
# ---------------------------------------------------------------------
with tab3:
    st.markdown("### Model Explainability (SHAP Features)")
    
    try:
        top_features = get_top_fraud_features(conn, limit=10)
        
        if not top_features.empty:
            st.markdown("#### Top Fraud-Driving Feature Combinations")
            
            # Bar chart
            top_features['feature_combo'] = (
                top_features['primary_feature'] + ' + ' + top_features['secondary_feature']
            )
            
            fig = px.bar(
                top_features.head(10),
                x='occurrence_count',
                y='feature_combo',
                orientation='h',
                title='Most Common Feature Combinations in Fraud Cases',
                labels={'occurrence_count': 'Occurrence Count', 'feature_combo': 'Feature Combination'},
                color='avg_fraud_score',
                color_continuous_scale='Reds',
                text='occurrence_count'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table with styling
            st.markdown("#### Feature Statistics")
            display_features = top_features.copy()
            display_features['avg_fraud_score'] = display_features['avg_fraud_score'].apply(lambda x: f"{x:.4f}")
            display_features['avg_amount'] = display_features['avg_amount'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(display_features, use_container_width=True, hide_index=True)
            
            # Key insights
            st.markdown("#### Key Insights")
            
            top_primary = top_features.iloc[0]
            st.info(
                f"**Primary Fraud Driver:** `{top_primary['primary_feature']}` "
                f"appears in **{top_primary['occurrence_count']}** fraud cases "
                f"with average score **{top_primary['avg_fraud_score']:.4f}** "
                f"and average amount **${top_primary['avg_amount']:,.2f}**"
            )
            
            if len(top_features) > 1:
                # Compare top 2
                second = top_features.iloc[1]
                st.success(
                    f"When combined with `{top_primary['secondary_feature']}`, "
                    f"the fraud signal is strongest. "
                    f"Secondary pattern: `{second['primary_feature']}` + `{second['secondary_feature']}` "
                    f"({second['occurrence_count']} cases)"
                )
        
        else:
            st.warning("No feature data available")
    
    except Exception as e:
        st.error(f"Error loading explainability data: {e}")
        import traceback
        st.code(traceback.format_exc())

# ---------------------------------------------------------------------
# TAB 4: PERFORMANCE METRICS
# ---------------------------------------------------------------------
with tab4:
    st.markdown("### System Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Fraud Score Distribution")
        try:
            score_dist = get_score_distribution(conn)
            
            if not score_dist.empty:
                fig = px.histogram(
                    score_dist,
                    x='fraud_score',
                    y='count',
                    title='Distribution of Fraud Scores',
                    labels={'fraud_score': 'Fraud Score', 'count': 'Transaction Count'},
                    nbins=50
                )
                
                # Add threshold line
                fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                             annotation_text="Threshold (0.5)")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No score distribution data")
        
        except Exception as e:
            st.error(f"Error loading score distribution: {e}")
    
    with col2:
        st.markdown("#### Processing Performance")
        try:
            # Query latency statistics
            latency_query = """
                SELECT 
                    COUNT(*) as total,
                    ROUND(AVG(processing_time_ms), 2) as avg_latency,
                    ROUND(MIN(processing_time_ms), 2) as min_latency,
                    ROUND(MAX(processing_time_ms), 2) as max_latency,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY processing_time_ms) as p50,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY processing_time_ms) as p95,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY processing_time_ms) as p99
                FROM fraud_decisions
                WHERE processing_time_ms > 0
            """
            
            latency_stats = pd.read_sql(latency_query, conn)
            
            if not latency_stats.empty and latency_stats['total'].iloc[0] > 0:
                stats = latency_stats.iloc[0]
                
                st.metric("Total Decisions", f"{int(stats['total']):,}")
                st.metric("Avg Latency", f"{stats['avg_latency']:.2f} ms")
                st.metric("Min Latency", f"{stats['min_latency']:.2f} ms")
                st.metric("Max Latency", f"{stats['max_latency']:.2f} ms")
                st.metric("P50 (Median)", f"{stats['p50']:.2f} ms")
                st.metric("P95", f"{stats['p95']:.2f} ms")
                st.metric("P99", f"{stats['p99']:.2f} ms")
                
                # Performance gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=stats['avg_latency'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Avg Latency (ms)"},
                    delta={'reference': 5},
                    gauge={
                        'axis': {'range': [None, 20]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 5], 'color': "lightgreen"},
                            {'range': [5, 10], 'color': "yellow"},
                            {'range': [10, 20], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 10
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No performance data available")
        
        except Exception as e:
            st.error(f"Error loading performance metrics: {e}")

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Real-Time Fraud Detection Dashboard | Powered by PostgreSQL + XGBoost + SHAP</p>
        <p>Data refreshes automatically | Connected to streaming pipeline</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Note: Connection is cached and will be reused, no need to close
