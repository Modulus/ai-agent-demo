from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
import sys
import yfinance as yf

def get_stock_price(ticker: str):
    try:
        return yf.download(ticker, period="1d")
    except Exception as e:
        print(f"Failed to get stock price for ticker, because error: {e}", file=sys.stderr)
        return None   


agent = Agent('ollama:llama3.2', result_type=str)

result = agent.run_sync('What is the stock ticker of Apple?')
print(result.data)

print(get_stock_price("TSLA"))

print(get_stock_price("AAPL"))