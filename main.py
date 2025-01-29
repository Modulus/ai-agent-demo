from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
import sys
import yfinance as yf


class StockPrice(BaseModel):
    stock_symbol: str = Field(alias="stock_ticker", description="The stock ticker symbol of the stock, abbreviated name of the stock")
    company_name: str = Field(alias="company_name", description="The name of the company that the stock belongs to")
    close: float = Field(alias="close", description="The closing price of the stock, meaning the price when the stock market closed")
    high: float = Field(alias="high", description="The highest price of the stock has has during the trading day")
    low: float = Field(alias="low", description="The lowest price of the stock has has during the trading day")
    open: float = Field(description="The opening price of the stock, meaning the price when the stock marked opened")
    volume: int = Field(description="The trading volume of the stock traded will allways be a whole number")

class StockTickerSymbol(BaseModel):
    stock_ticker: str
    company_name: str





def get_stock_price(ctx: RunContext[None]) -> StockPrice:
    try:
        ticker_result = ticker_agent.run_sync('What is the stock ticker of Tesla?', usage_limits=UsageLimits(request_limit=5, total_tokens_limit=1000))
        print(f" Got data from agent: {ticker_result.data}")
        return yf.download(ticker_result.data.stock_ticker, period="1d")
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
    result_type=str,
    system_prompt="You are an agent that can provide stock prices for companies. \
    Please provide the stock price for the company you are asked about. using the stock ticker symbol you have been provided and the tools at your disposal.",
    tools=[get_stock_price])
    


print("Using ticker for next agent query")

price_result = stock_price_agent.run_sync(
    "What is the stock Price for Tesla at the moment use the from the ticker_result?",
    usage_limits=UsageLimits(request_limit=5, total_tokens_limit=1000)
    )
print(price_result.data)