from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
import sys
import yfinance as yf


class StockPrice(BaseModel):
    stock_symbol: str
    company_name: str
    close: float
    high: float
    low: float
    open: float
    volume: int

class StockTickerSymbol(BaseModel):
    stock_ticker: str
    company_name: str

 # Setup tools
 # @stock_price_agent.tool_plain
def get_stock_price(ticker_meta: StockTickerSymbol):
    try:
        return yf.download(ticker_meta.stock_ticker, period="1d")
    except Exception as e:
        print(f"Failed to get stock price for ticker, because error: {e}", file=sys.stderr)
        return None   



# Setup agents
ticker_agent = Agent(
    'ollama:llama3.2', 
    result_type=StockTickerSymbol,
    system_prompt="You are an agent that can provide stock ticker symbols for companies. \
        Please provide the stock ticker symbol for the company you are asked about.")

stock_price_agent = Agent(
    'ollama:llama3.2', 
    result_type=StockPrice,
    system_prompt="You are an agent that can provide stock prices for companies. \
    Please provide the stock price for the company you are asked about. using the stock ticker symbol you have been provided and the tools at your disposal.",
    tools=[get_stock_price])
    

ticker_result = ticker_agent.run_sync('What is the stock ticker of Tesla?')
print(ticker_result.data)


print("Using ticker for next agent query")

price_result = stock_price_agent.run_sync(f'What is the stock Price for Tesla at the moment use the ticker?', deps=[ticker_result.data])
print(price_result.data)