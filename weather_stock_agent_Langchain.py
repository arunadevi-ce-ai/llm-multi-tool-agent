import os
from dotenv import load_dotenv
load_dotenv()   # ← must run BEFORE ChatOpenAI() is created
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import requests

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = requests.get(geo_url, params={"name": city, "count": 1}).json()
    if "results" not in geo_resp:
        return f"Could not find location: {city}"
    lat = geo_resp["results"][0]["latitude"]
    lon = geo_resp["results"][0]["longitude"]
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_resp = requests.get(weather_url, params={
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m"
    }).json()
    temp = weather_resp["current"]["temperature_2m"]
    humidity = weather_resp["current"]["relative_humidity_2m"]
    return f"{temp}°C, {humidity}% humidity"


@tool
def get_stock_price(ticker: str) -> str:
    """Get current stock price for a given ticker symbol."""
    url = "https://www.alphavantage.co/query"
    resp = requests.get(url, params={
        "function": "GLOBAL_QUOTE",
        "symbol": ticker.upper(),
        "apikey": os.environ.get("ALPHA_VANTAGE_API_KEY")
    }).json()
    quote = resp.get("Global Quote", {})
    price = quote.get("05. price")
    if not price:
        return f"Could not find price for ticker: {ticker}"
    return f"${float(price):.2f}"


llm = ChatOpenAI(model="gpt-5.4-mini")
tools = [get_weather, get_stock_price]

agent = create_agent(llm, tools)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Singapore and the stock price of TSLA?"}]
})

print(result["messages"][-1].content)
# print(result["messages"])