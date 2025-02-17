import yfinance as yf

ril = yf.Ticker('RELIANCE.NS')
ril_news = ril.news()
print(ril_news)