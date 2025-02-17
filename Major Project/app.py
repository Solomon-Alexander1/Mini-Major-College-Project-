from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from textblob import TextBlob
import ta
import google.generativeai as genai

app = Flask(__name__)

genai.configure(api_key="AIzaSyA4sBUMWL9M0ddYlG2ZC7W85XhxzBxBtEk")
model = genai.GenerativeModel('gemini-pro')

# Analysis Techniques
technical_techniques = [
    'Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', 'Fibonacci Retracement',
    'Volume Analysis', 'Support and Resistance', 'Candlestick Patterns', 'Ichimoku Cloud', 'ADX'
]

fundamental_techniques = [
    'Earnings Analysis (EPS, P/E Ratio)', 'Revenue Growth', 'Debt-to-Equity Ratio',
    'Return on Equity (ROE)', 'Price-to-Book (P/B) Ratio', 'Free Cash Flow (FCF) Analysis',
    'Dividend Yield and Growth', 'Discounted Cash Flow (DCF) Model', 'Industry Analysis',
    'Macroeconomic Indicators (GDP, Unemployment, Interest Rates)'
]

sentiment_techniques = [
    'Social Media Sentiment Analysis (Twitter, Reddit)', 'News Sentiment Analysis', 'Put/Call Ratio',
    'Volatility Index (VIX)', 'Market Breadth Indicators', 'Short Interest Ratio', 'Consumer Confidence Index',
    'Analyst Sentiment', 'Survey-Based Indicators', 'Implied Volatility in Options Markets'
]

# Mock Gemini API
def gemini(payload, ticker):
    i = f"Please provide a detailed analysis report for Ticker: {ticker}\nPlease generate a detailed analysis report based on the technical, fundamental, and sentiment analysis, understandable by a layman, without any recommendations and disclaimer based on all the publicly available and provided information."
    r = model.generate_content(i).text.replace("*", "")
    return r

# Fetch Real-Time Data
def fetch_realtime_data(ticker, timeframe):
    return yf.download(ticker, period="1d", interval=timeframe)

# Technical Analysis
def calculate_technical_analysis(df, techniques):
    analysis = {}

    # 1. Moving Averages (SMA)
    if 'Moving Averages' in techniques:
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        analysis['Moving Averages'] = {
            'SMA_50': df['SMA_50'].iloc[-1] if len(df) >= 50 else "Not enough data",
            'SMA_200': df['SMA_200'].iloc[-1] if len(df) >= 200 else "Not enough data"
        }

    # 2. RSI (Relative Strength Index)
    if 'RSI' in techniques:
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        analysis['RSI'] = df['RSI'].iloc[-1] if len(df) > 0 else "No data"

    # 3. MACD (Moving Average Convergence Divergence)
    if 'MACD' in techniques:
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()
        analysis['MACD'] = {
            'MACD': df['MACD'].iloc[-1],
            'MACD_Signal': df['MACD_signal'].iloc[-1],
            'MACD_Histogram': df['MACD_histogram'].iloc[-1]
        }

    # 4. Bollinger Bands
    if 'Bollinger Bands' in techniques:
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        analysis['Bollinger Bands'] = {
            'Upper Band': df['BB_upper'].iloc[-1],
            'Lower Band': df['BB_lower'].iloc[-1],
            'Middle Band': df['BB_middle'].iloc[-1]
        }

    # 5. Fibonacci Retracement (approximately)
    if 'Fibonacci Retracement' in techniques:
        max_price = df['High'].max()
        min_price = df['Low'].min()
        difference = max_price - min_price
        analysis['Fibonacci Retracement'] = {
            'Level_0': max_price,
            'Level_23.6': max_price - difference * 0.236,
            'Level_38.2': max_price - difference * 0.382,
            'Level_50': max_price - difference * 0.5,
            'Level_61.8': max_price - difference * 0.618,
            'Level_100': min_price
        }

    # 6. Volume Analysis
    if 'Volume Analysis' in techniques:
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        analysis['Volume Analysis'] = {
            'Volume_20_MA': df['Volume_MA_20'].iloc[-1] if len(df) >= 20 else "Not enough data"
        }

    # 7. Support and Resistance (approximation)
    if 'Support and Resistance' in techniques:
        max_price = df['High'].max()
        min_price = df['Low'].min()
        analysis['Support and Resistance'] = {
            'Support': min_price,
            'Resistance': max_price
        }

    # 8. Candlestick Patterns (examples)
    if 'Candlestick Patterns' in techniques:
        df['Hammer'] = ta.patterns.cdl_hammer(df['Open'], df['High'], df['Low'], df['Close'])
        df['Engulfing'] = ta.patterns.cdl_engulfing(df['Open'], df['High'], df['Low'], df['Close'])
        analysis['Candlestick Patterns'] = {
            'Hammer': df['Hammer'].iloc[-1],
            'Engulfing': df['Engulfing'].iloc[-1]
        }

    # 9. Ichimoku Cloud
    if 'Ichimoku Cloud' in techniques:
        ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'], df['Close'])
        df['Ichimoku_A'] = ichimoku.ichimoku_a()
        df['Ichimoku_B'] = ichimoku.ichimoku_b()
        analysis['Ichimoku Cloud'] = {
            'Ichimoku_A': df['Ichimoku_A'].iloc[-1],
            'Ichimoku_B': df['Ichimoku_B'].iloc[-1]
        }

    # 10. ADX (Average Directional Index)
    if 'ADX' in techniques:
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx.adx()
        df['ADX_positive'] = adx.adx_pos()
        df['ADX_negative'] = adx.adx_neg()
        analysis['ADX'] = {
            'ADX': df['ADX'].iloc[-1],
            'ADX_Positive': df['ADX_positive'].iloc[-1],
            'ADX_Negative': df['ADX_negative'].iloc[-1]
        }

    return analysis

# Fundamental Analysis
def calculate_fundamental_analysis(ticker, techniques):
    stock = yf.Ticker(ticker)
    info = stock.info
    analysis = {}
    if 'Earnings Analysis (EPS, P/E Ratio)' in techniques:
        analysis['EPS'] = info.get('trailingEps', 'N/A')
        analysis['P/E Ratio'] = info.get('trailingPE', 'N/A')
    if 'Revenue Growth' in techniques and 'revenueGrowth' in info:
        analysis['Revenue Growth'] = f"{info['revenueGrowth'] * 100:.2f}%" if info['revenueGrowth'] else "N/A"
    return analysis

# Sentiment Analysis
def calculate_sentiment_analysis(ticker, techniques):
    stock = yf.Ticker(ticker)
    news = stock.news
    sentiment = {}
    if news:
        headlines = ' '.join([n['title'] for n in news])
        polarity = TextBlob(headlines).sentiment.polarity
        sentiment['Overall Sentiment'] = (
            'Positive' if polarity > 0 else
            'Negative' if polarity < 0 else 'Neutral'
        )
        sentiment['News Headlines'] = [n['title'] for n in news[:5]]  # Show first 3 headlines
    else:
        sentiment['Overall Sentiment'] = "Neutral (No recent news)"
        sentiment['News Headlines'] = []
    return sentiment

@app.route('/')
def index():
    return render_template('index.html',
                           technical_techniques=technical_techniques,
                           fundamental_techniques=fundamental_techniques,
                           sentiment_techniques=sentiment_techniques)

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form['ticker']
    timeframe = request.form['timeframe']
    selected_technical = request.form.getlist('technical')
    selected_fundamental = request.form.getlist('fundamental')
    selected_sentiment = request.form.getlist('sentiment')

    # Validate ticker
    if not ticker:
        return "Ticker symbol is required!", 400

    # Fetch stock data based on the selected timeframe
    try:
        df = yf.download(ticker, period="6mo", interval=timeframe)
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}", 500

    # Perform analyses
    technical_analysis = calculate_technical_analysis(df, selected_technical)
    fundamental_analysis = calculate_fundamental_analysis(ticker, selected_fundamental)
    sentiment_analysis = calculate_sentiment_analysis(ticker, selected_sentiment)

    # Mock Gemini API Call
    gemini_payload = {
        'technical': technical_analysis,
        'fundamental': fundamental_analysis,
        'sentiment': sentiment_analysis
    }
    gemini_response = gemini(gemini_payload, ticker)

    return render_template('result.html',
                           ticker=ticker,
                           timeframe=timeframe,
                           technical_analysis=technical_analysis,
                           fundamental_analysis=fundamental_analysis,
                           sentiment_analysis=sentiment_analysis,
                           overall_sentiment=gemini_response,
                           sentiment_news=sentiment_analysis['News Headlines'])

if __name__ == '__main__':
    app.run(debug=True)
