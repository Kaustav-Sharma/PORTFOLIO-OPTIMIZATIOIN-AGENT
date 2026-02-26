The Open Universe Wealth Agent operates on a four-phase quantitative pipeline, seamlessly bridging real-time data ingestion, natural language processing, genetic optimization, and human governance.
Phase 1: Data Ingestion & NLP Sentiment (data_engine.py)

The system bypasses static, backward-looking CSV datasets by utilizing a live data pipeline.

    Market Data Retrieval: Uses yfinance to fetch real-time pricing, dividend yields, and average trading volumes for any user-defined asset.

    FinBERT Sentiment Scoring: For each asset, the engine pulls the latest financial news headlines and processes them through a Hugging Face transformer model (ProsusAI/finbert). This generates a normalized sentiment score (ranging from negative to positive) that quantifies current market qualitative health.

    Microstructure Modeling: Calculates a custom "Liquidity Score" based on average volume, translating it into an estimated bid-ask spread (in basis points) to account for True Execution Cost (slippage).

Phase 2: Evolutionary Optimization (optimization_engine.py)

The mathematical core of the agent relies on the Non-dominated Sorting Genetic Algorithm II (NSGA-II), implemented via the DEAP framework, to navigate the non-convex search space of portfolio weights.

    Evolution Parameters: The engine initializes a population of 60 candidate portfolios and evolves them over 50 generations to find the Pareto-optimal frontier.

    Multi-Objective Fitness Function: The algorithm simultaneously evaluates three competing metrics:

        Maximize Net Yield: (Portfolio Yield) - (Execution Cost) + (Sentiment Score * Alpha)

        Minimize Risk: Measured as annualized historical volatility.

        Minimize Duration Gap: Aligning the portfolio's weighted duration with the user's target timeline.

    Dynamic Penalty System: To ensure 100% compliance with user constraints (e.g., Max Risk = 20%, Max Concentration = 40%), the engine applies severe mathematical deductions to the fitness score of any portfolio that breaches these boundaries, naturally weeding them out of the evolutionary gene pool.

Phase 3: Human-in-the-Loop Governance (app.py)

To satisfy institutional compliance and safety requirements, the AI is not granted direct execution authority.

    Stateful Orchestration: Built on Streamlit, the UI acts as the state manager, passing user constraints to the optimization engine and catching the resulting trade plan.

    Mandatory Review (HITL): The system generates a visual dashboard detailing the required "BUY/SELL" actions, exact share counts, and capital allocations. A human portfolio manager must review these and input manual "Manager Notes."

    Audit Trailing: Upon clicking "Approve," the virtual execution is logged into a permanent, timestamped Ledger dictionary (exportable as a CSV), recording the exact quantitative state of the transaction.

Phase 4: Walk-Forward Validation (backtest_engine.py)

To prove the efficacy of the sentiment-adjusted genetic algorithm, the system includes a robust historical testing module.

    Point-in-Time Testing: Users can select a past date via the "Time Machine" feature. The engine will artificially restrict its knowledge to data available prior to that date, generate target weights, and track performance up to the present day.

    Comparative Metrics: Evaluates the AI's optimized portfolio against a standard Equal-Weight benchmark, generating a comparative Matplotlib chart and calculating key metrics including Cumulative Return, Annualized Volatility, Max Drawdown, and the Sharpe Ratio.
Run the application :   
streamlit run app.py
