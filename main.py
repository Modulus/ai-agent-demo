from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
# import pyfinance as yf


class StockTicker(BaseModel):
    stock_symbol: str
    stock_price: float
    company_name: str



stock_price_agent = Agent('ollama:llama3.2', result_type=StockTicker)

# @stock_price_agent.tool_plain
# def get_stock_price(stock_symbol: str):
#     yf.

result = stock_price_agent.run_sync('What is the stock ticker of Apple?')
print(result.data)