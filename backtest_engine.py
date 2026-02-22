import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import datetime

def generate_backtest_report(tickers, target_weights, start_date):
    try:
        if start_date >= datetime.date.today():
            st.error("Cannot run a forward backtest starting from today. Select a past date.")
            return None, None, None

        data = yf.download(tickers, start=start_date, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close']
        else:
            close_prices = pd.DataFrame(data['Close'])
            close_prices.columns = tickers

        if close_prices.empty: 
            st.error("No historical data available for these tickers in the selected timeframe.")
            return None, None, None
        
        close_prices = close_prices.ffill().bfill()
        returns = close_prices.pct_change().fillna(0)
        
        port_returns = returns.dot(target_weights)
        port_cum = (1 + port_returns).cumprod()
        
        eq_weights = np.ones(len(tickers)) / len(tickers)
        bench_returns = returns.dot(eq_weights)
        bench_cum = (1 + bench_returns).cumprod()
        
        def get_metrics(ret_series, cum_series):
            cum_ret = (cum_series.iloc[-1] - 1) if not cum_series.empty else 0
            volatility = ret_series.std() * np.sqrt(252)
            sharpe = (ret_series.mean() / ret_series.std()) * np.sqrt(252) if ret_series.std() != 0 else 0
            
            cum_peaks = cum_series.cummax()
            drawdown = (cum_series - cum_peaks) / cum_peaks
            max_dd = drawdown.min() if not drawdown.empty else 0
            
            # --- NEW USEFUL STATS FOR THE REPORT ---
            win_rate = (ret_series > 0).mean()
            best_day = ret_series.max()
            worst_day = ret_series.min()
            
            return {
                "Cumulative Return": f"{cum_ret*100:.2f}%", 
                "Annual Volatility": f"{volatility*100:.2f}%", 
                "Sharpe Ratio": f"{sharpe:.2f}", 
                "Max Drawdown": f"{max_dd*100:.2f}%",
                "Daily Win Rate": f"{win_rate*100:.2f}%",
                "Best Day": f"{best_day*100:.2f}%",
                "Worst Day": f"{worst_day*100:.2f}%"
            }
            
        port_metrics = get_metrics(port_returns, port_cum)
        bench_metrics = get_metrics(bench_returns, bench_cum)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(port_cum.index, port_cum.values, label=f'AI Agent Portfolio', linewidth=2, color='#1f77b4')
        ax.plot(bench_cum.index, bench_cum.values, label=f'Equal Weight Benchmark', linewidth=2, color='gray', linestyle='--')
        
        ax.set_title(f'Walk-Forward Performance (Since {start_date})', fontsize=14, fontweight='bold')
        ax.set_ylabel('Growth of $1.00', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        return port_metrics, bench_metrics, fig
        
    except Exception as e:
        st.error(f"Backtest Engine Error: {str(e)}")
        return None, None, None