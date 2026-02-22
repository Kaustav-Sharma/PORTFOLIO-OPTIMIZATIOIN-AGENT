import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from transformers import pipeline
import datetime

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def fetch_live_news_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news
        if not news: return 0.0
        
        headlines = []
        for article in news[:8]:
            title = article.get('title') or article.get('content', {}).get('title', '')
            if title:
                headlines.append(title)
                
        if not headlines: 
            return 0.0
            
        analyzer = load_sentiment_model()
        results = analyzer(headlines)
        
        score = 0.0
        for res in results:
            if res['label'] == 'positive': score += res['score']
            elif res['label'] == 'negative': score -= res['score']
            
        return score / len(headlines)
    except Exception as e:
        print(f"Sentiment Error for {ticker}: {e}")
        return 0.0 

def estimate_duration(ticker, info):
    bond_keywords = ['treasury', 'bond', 'corporate', 'yield', 'aggregate']
    name = info.get('longName', '').lower()
    category = info.get('category', '').lower()
    
    if any(k in name or k in category for k in bond_keywords):
        if 'short' in name: return 2.0
        if 'intermediate' in name: return 7.0
        if 'long' in name or '20+' in name: return 18.0
        return 5.0
    return 0.0

@st.cache_data(ttl=300)
def fetch_single_asset(ticker, as_of_date=None):
    try:
        obj = yf.Ticker(ticker)
        info = obj.info
        
        is_historical = as_of_date is not None and as_of_date < datetime.date.today()
        
        if is_historical:
            end_dt = pd.to_datetime(as_of_date)
            start_dt = end_dt - pd.Timedelta(days=365)
            hist = obj.history(start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'))
            
            if hist.empty: return None
            
            price = hist['Close'].iloc[-1]
            rets = hist['Close'].pct_change().dropna()
            vol = rets.std() * np.sqrt(252) if not rets.empty else 0.20
            
            # HISTORICAL SENTIMENT PROXY
            try:
                hist_7d = hist.tail(7)
                if not hist_7d.empty and len(hist_7d) > 1:
                    ret_7d = (hist_7d['Close'].iloc[-1] / hist_7d['Close'].iloc[0]) - 1
                    sentiment_score = max(min(ret_7d * 15, 1.0), -1.0)
                else:
                    sentiment_score = 0.0
            except:
                sentiment_score = 0.0
                
        else:
            # LIVE DATA
            price = info.get('currentPrice') or info.get('previousClose')
            if price is None:
                hist = obj.history(period="5d")
                if not hist.empty: price = hist['Close'].iloc[-1]
            
            hist = obj.history(period="1y")
            if not hist.empty:
                rets = hist['Close'].pct_change().dropna()
                vol = rets.std() * np.sqrt(252)
            else:
                vol = 0.20
                
            sentiment_score = fetch_live_news_sentiment(ticker)
            
            # WEEKEND/ETF FALLBACK: If there are no news headlines, infer sentiment from the last 7 days of price action
            if sentiment_score == 0.0:
                try:
                    hist_7d = hist.tail(7)
                    if not hist_7d.empty and len(hist_7d) > 1:
                        ret_7d = (hist_7d['Close'].iloc[-1] / hist_7d['Close'].iloc[0]) - 1
                        sentiment_score = max(min(ret_7d * 15, 1.0), -1.0)
                except:
                    pass
            
        if price is None: return None
            
        yld = info.get('yield', 0) or info.get('dividendYield', 0) or 0.0
        avg_vol = info.get('averageVolume', 1_000_000)
        liq_score = min(avg_vol / 500_000, 1.0)
        spread = 2.0 + (1.0 - liq_score) * 20
        
        return {
            "Ticker": ticker.upper(),
            "Name": info.get('longName', ticker),
            "Price": price,
            "Yield": yld,
            "Risk_Vol": vol,
            "Duration": estimate_duration(ticker, info),
            "Liquidity": liq_score,
            "Spread_bps": spread,
            "Sentiment": sentiment_score
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def update_market_data(tickers, as_of_date=None):
    data = []
    for t in tickers:
        asset_data = fetch_single_asset(t, as_of_date)
        if asset_data:
            data.append(asset_data)
    return pd.DataFrame(data)