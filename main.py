from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
import sys
import yfinance as yf
from rich.prompt import Prompt

class StockPrice(BaseModel):
    price: float = Field(alias="stock price", description="The closing price of the stock, meaning the price when the stock market closed")


class StockTickerSymbol(BaseModel):
    stock_ticker: str = Field(alias="stock_ticker", description="The stock ticker symbol of the stock, abbreviated name")
    company_name: str = Field(alias="company_name", description="The name of the company that the stock belongs to")





def get_stock_price(ctx: RunContext[StockTickerSymbol]) -> str:
    try:

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
    Please provide the stock price for the company you are asked about. using the stock ticker symbol you have been provided. Use the get_stock_price function to get the stock price, and do not generate code to do so",
    tools=[get_stock_price],
    end_strategy="exhaustive", 
    result_retries=5)
    

prompt =Prompt.ask("What company would you like to know the stock price for?")
print(f"Using company name: {prompt} for next agent query")
ticker_result = ticker_agent.run_sync(f"What is the stock ticker of {prompt}?", usage_limits=UsageLimits(request_limit=5, total_tokens_limit=1000))
print(f"Found ticker {ticker_result.data}")

if not ticker_result.data:
    print("No data from agent")

else:
    print(f" Got data from agent: {ticker_result.data}")

    price_result = stock_price_agent.run_sync(
        f"What is the stock Price for {ticker_result} at the moment use the from",
        usage_limits=UsageLimits(request_limit=5, total_tokens_limit=3000)
        )
    print(price_result.data)