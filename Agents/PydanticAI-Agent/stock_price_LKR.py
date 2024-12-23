from pydantic_ai import Agent
from pydantic import BaseModel
import yfinance as yf
import requests

class StockPriceResult(BaseModel):
    symbol: str
    price: float
    currency: str
    message: str

stock_agent = Agent(
    "groq:llama3-groq-70b-8192-tool-use-preview",
    result_type=StockPriceResult,
    system_prompt="You are a helpful financial assistant that can look up stock prices and convert them to LKR in real-time. Use the get_stock_price tool to fetch stock data and the convert_to_lkr tool to convert the price."
)

@stock_agent.tool_plain
def get_stock_price(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    price = ticker.fast_info.last_price
    return {
        "symbol": symbol.upper(),
        "price": round(price, 2),
        "currency": "USD"
    }

@stock_agent.tool_plain
def convert_to_lkr(amount: float, currency: str) -> dict:
    if currency.upper() == "USD":
        # Fetch exchange rate from Exchangerate API
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        data = response.json()
        exchange_rate = data.get("rates", {}).get("LKR", 1)  # Default to 1 if unavailable
        converted_price = round(amount * exchange_rate, 2)
        return {
            "converted_price": converted_price,
            "currency": "LKR"
        }
    return {
        "converted_price": amount,
        "currency": currency
    }

# Run the agent
result = stock_agent.run_sync("What is Apple's current stock price?")
print(f"Stock Price: {result.data.price:.2f} {result.data.currency}")
print(f"Message: {result.data.message}")
