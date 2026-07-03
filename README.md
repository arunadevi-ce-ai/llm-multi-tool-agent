# Multi-Tool LLM Agent

A hand-built example of LLM tool-calling: a single OpenAI-powered agent that can look up **live weather** and **real-time stock prices**, choosing which tool(s) to use based on the user's question, then combining the results into one natural-language answer.

Built as a learning project to understand the core "agentic AI" pattern — model decides → code executes → model responds — without relying on a framework, before moving on to LangChain.

## What it does

Ask a question like:
> "What's the weather in Singapore and the stock price of TSLA?"

The agent:
1. Sends the question to the LLM along with a list of available tools
2. The model decides which tool(s) are needed and with what arguments (it can call **multiple tools in a single turn**)
3. The code executes the real API calls (Open-Meteo for weather, Alpha Vantage for stock prices)
4. The tool results are sent back to the model, which produces one combined, natural-language answer

Example output:
```
Tool called: get_weather {'city': 'Singapore'}
Tool result: 29.5°C, 71% humidity
Tool called: get_stock_price {'ticker': 'TSLA'}
Tool result: $393.45
The weather in Singapore is 29.5°C with 71% humidity, and TSLA is trading at $393.45.
```

## Why I built this

I wanted to understand tool-calling at the mechanical level — what the API actually sends and expects — before relying on a framework like LangChain to automate it. This project deliberately does **not** use an agent framework, so every step (routing tool calls, matching `call_id`s, handling multiple simultaneous tool calls, sending results back for a final answer) is visible and understood, not abstracted away.

## Tech stack

- **OpenAI API** (`responses.create`) — function/tool calling
- **Open-Meteo API** — free, no-key weather + geocoding
- **Alpha Vantage API** — free-tier real-time stock quotes
- Python, `requests`, `python-dotenv`

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/llm-multi-tool-agent.git
   cd llm-multi-tool-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your own keys:
   ```bash
   cp .env.example .env
   ```
   - Get an OpenAI key at https://platform.openai.com
   - Get a free Alpha Vantage key at https://www.alphavantage.co/support/#api-key

4. Run it:
   ```bash
   python weather_stock_agent.py
   ```

## Design notes / things I learned

- **Tool descriptions matter as much as prompts** — the model never sees your code, only the JSON schema description of each tool, so vague descriptions lead to wrong tool selection.
- **`call_id` matching** is what lets the model correctly attribute a tool's result back to the specific call it made — essential once more than one tool is called in a single turn.
- **Alpha Vantage's free tier is rate-limited** (~1 request/sec in practice). Hitting it too fast returns a `"Note"` field instead of real data rather than an obvious error — this is handled explicitly in the code rather than silently failing.
- Handling the "no tool needed" case (e.g. a question needing no data lookup) is easy to forget and will break naive code that assumes every response is a tool call.

## Possible next steps

- [ ] Rebuild this using LangChain's `@tool` decorator and agent executor, and compare against this manual version
- [ ] Add caching to avoid redundant API calls / rate-limit issues
- [ ] Add a third tool connected to cloud infrastructure (e.g. checking an AWS service status or a GitHub repo's latest commit)
- [ ] Wrap as an interactive CLI loop with persistent conversation memory

## About me

Cloud/DevOps engineer (AWS, GCP, Terraform, Kubernetes) learning to build and deploy agentic AI systems on top of existing infrastructure skills. [Add your LinkedIn / portfolio link here]
