"""
Multi-tool LLM Agent Example
-----------------------------
This script demonstrates the core "agentic" pattern:
  1. Ask the model a question, along with a list of tools it's allowed to use.
  2. The model decides WHICH tool(s) to call and WITH WHAT arguments (it never
     runs code itself — it just tells you what it wants).
  3. Your code actually executes the real function(s).
  4. You send the results back to the model so it can produce one final,
     natural-language answer that combines everything.

This is the same "plan -> act -> observe -> respond" loop that frameworks
like LangChain later automate for you.
"""

import time
import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from your .env file (OPENAI_API_KEY, ALPHA_VANTAGE_API_KEY)
# into the environment so os.environ.get(...) can find them.
# NOTE: this must run BEFORE you try to read any environment variables below.
load_dotenv()

# Create the OpenAI client using the key loaded from .env.
# Never hardcode API keys directly in code — they end up in Git history
# and screenshots if you do.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# TOOL 1: Weather lookup (real API — Open-Meteo, no API key required)
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """
    Given a city name, return current temperature and humidity as a string.

    Open-Meteo's forecast API needs latitude/longitude, not a city name,
    so this function does two API calls:
      Step 1: geocode the city name -> lat/lon
      Step 2: fetch current weather for those coordinates
    """
    # --- Step 1: Convert the city name into coordinates ---
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = requests.get(geo_url, params={"name": city, "count": 1}).json()

    # If the geocoding API found nothing, "results" won't be in the response.
    # Always guard against this — the model may pass a misspelled or unknown city.
    if "results" not in geo_resp:
        return f"Could not find location: {city}"

    lat = geo_resp["results"][0]["latitude"]
    lon = geo_resp["results"][0]["longitude"]

    # --- Step 2: Fetch current weather using those coordinates ---
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_resp = requests.get(weather_url, params={
        "latitude": lat,
        "longitude": lon,
        # "current" tells the API which live fields we want back
        "current": "temperature_2m,relative_humidity_2m"
    }).json()

    temp = weather_resp["current"]["temperature_2m"]
    humidity = weather_resp["current"]["relative_humidity_2m"]

    # We return a plain string — the model doesn't need structured data back,
    # it just needs enough text to build a natural-language sentence with.
    return f"{temp}°C, {humidity}% humidity"


# ---------------------------------------------------------------------------
# TOOL 2: Stock price lookup (real API — Alpha Vantage, free tier)
# ---------------------------------------------------------------------------

# Alpha Vantage requires a free API key (from alphavantage.co) stored in .env
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")


def get_stock_price(ticker: str) -> str:
    """
    Given a stock ticker (e.g. "TSLA"), return the current price as a string.

    NOTE: Alpha Vantage's free tier is rate-limited (~1 request/sec in
    practice, officially 5/min). If you call this function twice quickly,
    the second call can come back empty. If you see intermittent
    "Could not find price" errors, that's very likely the rate limit,
    not a bad ticker — add a short delay (e.g. time.sleep(2)) before the
    request if you're calling this function back-to-back.
    """
    url = "https://www.alphavantage.co/query"
    resp = requests.get(url, params={
        "function": "GLOBAL_QUOTE",
        "symbol": ticker.upper(),
        "apikey": ALPHA_VANTAGE_KEY
    }).json()

    # Uncomment this line if you need to debug what Alpha Vantage actually
    # returned (e.g. to confirm a rate-limit message vs. a real bad ticker):
    # print("DEBUG raw response:", resp)

    quote = resp.get("Global Quote", {})
    price = quote.get("05. price")

    if not price:
        return f"Could not find price for ticker: {ticker}"

    return f"${float(price):.2f}"


# ---------------------------------------------------------------------------
# TOOL DEFINITIONS — describing our Python functions to the LLM
# ---------------------------------------------------------------------------
# IMPORTANT: the model can NOT see your Python code. It only knows what a
# tool does based on the "description" and "parameters" you write here.
# This is why clear, specific descriptions matter — vague descriptions lead
# the model to pick the wrong tool or the wrong arguments.
tools = [
    {
        "type": "function",              # tells the API this is a custom function tool
        "name": "get_weather",           # must match the Python function name you call later
        "description": "Get current weather for a city",
        "parameters": {                  # JSON Schema describing the expected arguments
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    },
    {
        "type": "function",
        "name": "get_stock_price",
        "description": "Get current stock price for a given ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    }
]

# ---------------------------------------------------------------------------
# STEP 1: Send the user's question to the model, along with the tool list.
# The model will decide whether it needs to call one or more tools to
# answer, and will NOT answer directly if it thinks a tool is needed.
# ---------------------------------------------------------------------------
response = client.responses.create(
    model="gpt-5.4-mini",
    input="What's the weather in Singapore and the stock price of TSLA?",
    tools=tools,
)

# We keep a running "conversation" list — this is the full message history
# that we'll send back to the model in Step 3, so it has full context of
# what was asked, what it decided to call, and what the results were.
conversation = [
    {"role": "user", "content": "What's the weather in Singapore and the stock price of TSLA?"}
]

# Collect the "function_call_output" items separately, so we can add them
# to the conversation only after we've processed every tool call the model
# asked for (a single question can trigger MULTIPLE tool calls at once,
# as it does here — one for weather, one for stock price).
function_outputs = []

# ---------------------------------------------------------------------------
# STEP 2: Loop through everything the model returned.
# response.output is a LIST because the model can return multiple items in
# one turn — e.g. two separate function_call items for two separate tools.
# ---------------------------------------------------------------------------
for item in response.output:
    if item.type == "function_call":
        # item.arguments comes back as a JSON STRING (e.g. '{"city": "Singapore"}'),
        # so we parse it into a real Python dict before using it.
        args = json.loads(item.arguments)

        # Route to the correct Python function based on which tool name
        # the model chose. As you add more tools, add more elif branches here.
        if item.name == "get_weather":
            result = get_weather(args["city"])
        elif item.name == "get_stock_price":
            result = get_stock_price(args["ticker"])
        else:
            result = "Unknown tool"

        print("Tool called:", item.name, args)
        print("Tool result:", result)

        # Add the model's own function-call request to the conversation
        # history — the API needs this so it remembers what it asked for.
        conversation.append(item)

        # Build the "answer" for this specific tool call. The call_id here
        # is what links this result back to the exact function_call the
        # model made — important once there are multiple calls in one turn.
        function_outputs.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": result
        })

    elif item.type == "message":
        # If the model didn't need any tool at all (e.g. for a question like
        # "tell me a joke"), it returns a plain text message instead.
        print(item.content[0].text)

# Add all of the tool results into the conversation, after all the
# function_call items. Order matters here: question -> calls -> results.
conversation.extend(function_outputs)

# ---------------------------------------------------------------------------
# STEP 3: Send the full conversation (question + tool calls + tool results)
# back to the model in a SECOND request. Now that the model can see the
# actual data, it can combine both results into one natural-language answer
# instead of you just printing raw tool output.
# ---------------------------------------------------------------------------
followup = client.responses.create(
    model="gpt-5.4-mini",
    input=conversation,
    tools=tools,
)

# output_text is the model's final, human-readable answer, e.g.:
# "The weather in Singapore is 29.5°C with 71% humidity, and TSLA is
#  trading at $393.45."
print(followup.output_text)
