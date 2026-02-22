import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import io 

st.set_page_config(page_title="Open Universe Wealth Agent", layout="wide")

from data_engine import update_market_data
from optimization_engine import run_rebalancing
from backtest_engine import generate_backtest_report

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "tickers" not in st.session_state: st.session_state["tickers"] = ["SPY", "TLT", "GLD"]
if "optimization_result" not in st.session_state: st.session_state["optimization_result"] = None
if "audit_log" not in st.session_state: st.session_state["audit_log"] = []
if "market_data" not in st.session_state: st.session_state["market_data"] = pd.DataFrame()
if "sim_date" not in st.session_state: st.session_state["sim_date"] = datetime.date.today()

# Dedicated memory for the Report tab so it survives trade approvals
if "report_weights" not in st.session_state: st.session_state["report_weights"] = None
if "report_tickers" not in st.session_state: st.session_state["report_tickers"] = None

# DECOUPLED STATE STORAGE
if "portfolio_holdings" not in st.session_state: 
    st.session_state["portfolio_holdings"] = {t: 0.0 for t in st.session_state["tickers"]}
if "portfolio_cash" not in st.session_state: 
    st.session_state["portfolio_cash"] = 10000.0

st.title("🌍 Open Universe Wealth Agent")
st.markdown("**Search & Add ANY Asset • Point-in-Time Optimization • Generate Reports**")
st.markdown("___")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("1. Build Your Universe")

new_ticker = st.sidebar.text_input("Add Ticker (e.g., NVDA, BTC-USD)", "").upper()
if st.sidebar.button("Add Asset"):
    if new_ticker and new_ticker not in st.session_state["tickers"]:
        st.session_state["tickers"].append(new_ticker)
        st.session_state["portfolio_holdings"][new_ticker] = 0.0
        st.session_state["market_data"] = pd.DataFrame() 
        st.rerun()

st.sidebar.subheader("Or Load Sector Packs:")
col_p1, col_p2 = st.sidebar.columns(2)
if col_p1.button("Tech Giants"):
    for t in ["AAPL", "MSFT", "NVDA", "GOOG"]:
        if t not in st.session_state["tickers"]: 
            st.session_state["tickers"].append(t)
            st.session_state["portfolio_holdings"][t] = 0.0
    st.session_state["market_data"] = pd.DataFrame() 
    st.rerun()

if col_p2.button("Crypto"):
    for t in ["BTC-USD", "ETH-USD"]:
        if t not in st.session_state["tickers"]: 
            st.session_state["tickers"].append(t)
            st.session_state["portfolio_holdings"][t] = 0.0
    st.session_state["market_data"] = pd.DataFrame() 
    st.rerun()

if st.sidebar.button("Clear Universe"):
    st.session_state["tickers"] = []
    st.session_state["portfolio_holdings"] = {}
    st.session_state["market_data"] = pd.DataFrame()
    st.session_state["report_weights"] = None 
    st.rerun()

# --- TIME MACHINE ---
st.sidebar.markdown("---")
st.sidebar.header("⏱️ Test on previous dates")
st.sidebar.caption("Select a past date. The AI will optimize using ONLY data prior to this date.")

today = datetime.date.today()
new_date = st.sidebar.date_input("Optimization As-Of Date", value=st.session_state["sim_date"], max_value=today)

if new_date > today:
    st.sidebar.error("Future dates are invalid. Resetting to today.")
    new_date = today
    time.sleep(1)
    st.rerun()

if new_date != st.session_state["sim_date"]:
    st.session_state["sim_date"] = new_date
    st.session_state["market_data"] = pd.DataFrame() 
    st.rerun()

if st.session_state["tickers"]:
    st.sidebar.markdown("### Active Assets")
    st.sidebar.write(", ".join(st.session_state["tickers"]))
    if st.sidebar.button("🔄 Force Data Refresh"):
        st.session_state["market_data"] = pd.DataFrame()
        st.rerun()

# ==========================================
# MAIN PAGE: TABS & DATA FETCHING
# ==========================================
if st.session_state["market_data"].empty and st.session_state["tickers"]:
    with st.spinner(f"Fetching Data as of {st.session_state['sim_date']}..."):
        st.session_state["market_data"] = update_market_data(st.session_state["tickers"], as_of_date=st.session_state["sim_date"])

if not st.session_state["market_data"].empty:
    df = st.session_state["market_data"]
    
    tab_live, tab_report = st.tabs(["⚙️ Agent Optimization & Execution", "📈 Comparison and Report"])
    
    with tab_live:
        df["Current_Shares"] = df["Ticker"].map(lambda t: st.session_state["portfolio_holdings"].get(t, 0.0))
        df["Current_Value"] = df["Current_Shares"] * df["Price"]
        total_equity = df["Current_Value"].sum() + st.session_state["portfolio_cash"]
        
        c_left, c_right = st.columns([1.5, 1])
        
        with c_left:
            st.markdown(f"### 📰 Market Data (As of {st.session_state['sim_date']})")
            display_df = df[["Ticker", "Name", "Price", "Sentiment", "Yield", "Risk_Vol"]].copy()
            display_df["Sentiment"] = display_df["Sentiment"].apply(lambda x: f"🟢 {x:.2f}" if x > 0.05 else (f"🔴 {x:.2f}" if x < -0.05 else f"⚪ {x:.2f}"))
            st.dataframe(display_df, use_container_width=True)

        with c_right:
            st.markdown("### 💼 Current Portfolio")
            
            holdings_data = []
            total_invested = 0.0
            
            for idx, row in df.iterrows():
                shares = row["Current_Shares"]
                if shares > 0:
                    val = row["Current_Value"]
                    total_invested += val
                    holdings_data.append({
                        "Asset": row["Ticker"],
                        "Shares": shares,
                        "Price": row["Price"],
                        "Total Value": val
                    })
            
            cash = max(0.0, st.session_state["portfolio_cash"])
            holdings_data.append({"Asset": "💵 CASH", "Shares": cash, "Price": 1.0, "Total Value": cash})
            
            holdings_df = pd.DataFrame(holdings_data)
            holdings_df["% Weight"] = (holdings_df["Total Value"] / total_equity) * 100
            
            st.dataframe(
                holdings_df.style.format({"Price": "${:,.2f}", "Total Value": "${:,.2f}", "% Weight": "{:.1f}%"}), 
                use_container_width=True, hide_index=True
            )
            st.metric("Total Account Equity", f"${total_equity:,.2f}")

        # --- MANUAL EDITING ---
        with st.expander("✏️ Edit Manual Holdings / Deposit Cash"):
            st.caption("Enter Holdings or deposit funds manually.")
            cols = st.columns(4)
            for i, row in df.iterrows():
                ticker = row['Ticker']
                with cols[i % 4]:
                    val = st.number_input(
                        f"{ticker} Shares", 
                        min_value=0.0, 
                        value=float(st.session_state["portfolio_holdings"].get(ticker, 0.0)), 
                        step=1.0, 
                        key=f"ui_hold_{ticker}"
                    )
                    st.session_state["portfolio_holdings"][ticker] = val
            
            cash_val = st.number_input(
                "Available Cash ($)", 
                min_value=0.0, 
                value=float(st.session_state["portfolio_cash"]), 
                step=1000.0, 
                key="ui_cash_input"
            )
            st.session_state["portfolio_cash"] = cash_val

        st.subheader("⚙️ Step 2: Optimization Strategy")
        c1, c2, c3 = st.columns(3)
        years = c1.slider("Timeline (Years)", 1, 30, 5)
        risk = c2.slider("Max Risk (Vol)", 0.05, 0.50, 0.20)
        conc = c3.slider("Max Concentration", 0.1, 1.0, 0.4)

        if st.button("🚀 Optimize Portfolio", type="primary"):
            if total_equity < 100:
                st.error("Equity too low.")
            else:
                with st.spinner("Running NSGA-II Agent..."):
                    res = run_rebalancing(df, total_equity, risk, 0.0, years, conc)
                    res["Trade_Shares"] = res["Target_Shares"] - res["Current_Shares"]
                    res["Trade_Value"] = res["Trade_Shares"] * res["Price"]
                    res["Action"] = res["Trade_Shares"].apply(lambda x: "BUY" if x > 0 else ("SELL" if x < 0 else "HOLD"))
                    
                    st.session_state["optimization_result"] = res
                    
                    # SAVE TO REPORT MEMORY
                    st.session_state["report_weights"] = res.set_index("Ticker")["Target_Weight"].reindex(df["Ticker"]).values
                    st.session_state["report_tickers"] = df["Ticker"].tolist()

        if st.session_state["optimization_result"] is not None:
            res = st.session_state["optimization_result"]
            
            st.markdown("### 📋 Trade Plan")
            trades = res[abs(res["Trade_Value"]) > 10].sort_values("Trade_Value", ascending=False)
            
            def color_action(val): return f'color: {"green" if val == "BUY" else "red" if val == "SELL" else "grey"}; font-weight: bold'
            st.dataframe(trades[["Ticker", "Price", "Current_Shares", "Target_Shares", "Action", "Trade_Value"]].style.applymap(color_action, subset=["Action"]).format({"Price": "${:.2f}", "Trade_Value": "${:,.2f}"}))

            st.markdown("___")
            st.markdown("### Human-in-the-Loop Approval")
            col_auth1, col_auth2 = st.columns([3, 1])
            with col_auth1:
                manager_notes = st.text_input("Manager Notes", placeholder="e.g., Authorized rebalance.")
            with col_auth2:
                st.write(""); st.write("")
                
                if st.button("✅ APPROVE TRADES", type="primary", use_container_width=True):
                    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    for _, row in trades.iterrows():
                        ticker = row["Ticker"]
                        trade_shares = row["Trade_Shares"]
                        trade_value = row["Trade_Value"]
                        
                        # CALC NEW VALUES
                        new_shares = max(0.0, float(st.session_state["portfolio_holdings"][ticker] + trade_shares))
                        new_cash = max(0.0, float(st.session_state["portfolio_cash"] - trade_value))
                        
                        # UPDATE DATA MEMORY ONLY
                        st.session_state["portfolio_holdings"][ticker] = new_shares
                        st.session_state["portfolio_cash"] = new_cash
                        
                        st.session_state["audit_log"].append({
                            "Date": ts, "Ticker": ticker, "Action": row["Action"],
                            "Shares": abs(trade_shares), "Price": row["Price"],
                            "Total Value": abs(trade_value), "Notes": manager_notes, "Status": "EXECUTED"
                        })
                        
                    st.success(f"Trades Executed and logged to Audit Trail!")
                    st.session_state["optimization_result"] = None # Hide the trade plan
                    time.sleep(1.5)
                    st.rerun() 
                    
                if st.button("🛑 REJECT TRADES", type="secondary", use_container_width=True):
                    st.error(f"Trades Rejected by Manager.")
                    st.session_state["optimization_result"] = None
                    time.sleep(1.5)
                    st.rerun()
                    
            if st.session_state["audit_log"]:
                st.markdown("#### 📖 Official Trade History")
                history_df = pd.DataFrame(st.session_state["audit_log"])
                def color_status(val):
                    color = 'green' if val == 'EXECUTED' else 'red'
                    return f'color: {color}; font-weight: bold'
                st.dataframe(history_df.style.applymap(color_status, subset=["Status"]), use_container_width=True)
                st.download_button("📥 Download Ledger (CSV)", data=history_df.to_csv(index=False).encode('utf-8'), file_name='portfolio_trade_history.csv', mime='text/csv')

    # ------------------------------------------
    # TAB 2: AUTO-GENERATED REPORT
    # ------------------------------------------
    with tab_report:
        st.markdown(f"### 📈 Performance Report (Walk-Forward Test)")
        
        if st.session_state['sim_date'] >= datetime.date.today():
            st.warning("⚠️ To generate a historical performance report, you must select a past 'Optimization As-Of Date' in the sidebar time machine!")
        elif st.session_state.get("report_weights") is None:
            st.info("⚠️ Please click 'Optimize Portfolio' in the Live tab to generate the target weights for the report.")
        else:
            st.write(f"Testing the AI's weight allocation from **{st.session_state['sim_date']}** up to **Today** against an Equal-Weight benchmark.")
            
            with st.spinner("Compiling Report..."):
                # Use the dedicated memory bank for the report
                target_weights = st.session_state["report_weights"]
                report_tickers = st.session_state["report_tickers"]
                
                port_metrics, bench_metrics, report_fig = generate_backtest_report(report_tickers, target_weights, st.session_state['sim_date'])
                
                if report_fig:
                    st.pyplot(report_fig)
                    st.markdown("---")
                    
                    m1, m2 = st.columns(2)
                    with m1: 
                        st.success("🤖 **AI Optimized Portfolio**")
                        st.json(port_metrics)
                    with m2: 
                        st.info("⚖️ **Equal-Weight Benchmark**")
                        st.json(bench_metrics)
                        
                    st.markdown("---")
                    st.markdown("### 📥 Export Report")
                    
                    buf = io.BytesIO()
                    report_fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                    buf.seek(0)
                    
                    metrics_df = pd.DataFrame({
                        "Metric": list(port_metrics.keys()),
                        "AI Portfolio": list(port_metrics.values()),
                        "Equal-Weight Benchmark": list(bench_metrics.values())
                    })
                    csv_data = metrics_df.to_csv(index=False).encode('utf-8')
                    
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            label="🖼️ Download Chart (PNG)",
                            data=buf,
                            file_name=f"AI_Portfolio_Chart_{st.session_state['sim_date']}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    with col_dl2:
                        st.download_button(
                            label="📊 Download Detailed Metrics (CSV)",
                            data=csv_data,
                            file_name=f"AI_Portfolio_Metrics_{st.session_state['sim_date']}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.error("Error generating report. Check if assets have enough historical data.")

else:
    st.info("👈 Use the Sidebar to add assets to your universe.")