# Bitcoin Signals Dashboard

A real-time Bitcoin trading signals dashboard that connects to Binance API and displays technical analysis indicators and trading signals.

## Features

- **Real-time Bitcoin Price**: Live BTC/USDT price updates from Binance
- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
- **Trading Signals**: Automated buy/sell signals based on technical analysis
- **Price Chart**: Real-time candlestick chart with 1-minute intervals
- **Responsive Design**: Works on desktop and mobile devices

## Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   npm install
   ```

3. Configure your Binance API keys in `.env` file (already configured)

4. Start the application:
   ```bash
   npm start
   ```

5. Open your browser and go to: `http://localhost:3000`

## Dashboard Components

### Price Card
- Current BTC/USDT price
- 24-hour price change percentage
- 24-hour volume

### Technical Indicators
- **RSI**: Values below 30 indicate oversold (potential buy), above 70 indicate overbought (potential sell)
- **MACD**: Momentum indicator for trend changes
- **Bollinger Bands**: Price volatility and potential support/resistance levels

### Signals
- **Buy Signals**: Green cards indicating potential buying opportunities
- **Sell Signals**: Red cards indicating potential selling opportunities
- **Signal Strength**: STRONG, MEDIUM, or WEAK based on indicator confidence

### Price Chart
- Real-time 1-minute candlestick data
- Updates every minute with latest price information

## Signal Logic

The dashboard generates signals based on:

1. **RSI Signals**:
   - Buy when RSI < 30 (oversold)
   - Sell when RSI > 70 (overbought)

2. **MACD Signals**:
   - Buy on bullish crossover (MACD line crosses above signal line)
   - Sell on bearish crossover (MACD line crosses below signal line)

3. **Bollinger Bands**:
   - Buy when price touches lower band
   - Sell when price touches upper band

## Data Updates

- Price data updates every 10 seconds via WebSocket
- Technical indicators and signals update every minute
- Chart data refreshes every minute

## Important Notes

- This is for educational purposes only
- Always do your own research before making trading decisions
- Past performance does not guarantee future results
- Consider risk management and position sizing

## API Keys Security

Your API keys are stored in the `.env` file and should never be shared or committed to version control. The current keys are configured for read-only access to market data.