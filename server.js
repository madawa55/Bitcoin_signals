const express = require('express');
const Binance = require('node-binance-api');
const WebSocket = require('ws');
const path = require('path');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3000;

const binance = Binance().options({
  APIKEY: process.env.BINANCE_API_KEY,
  APISECRET: process.env.BINANCE_API_SECRET,
  useServerTime: true,
  test: false
});

app.use(express.static('public'));
app.use(express.json());

let btcData = {
  price: 0,
  volume: 0,
  change24h: 0,
  signals: [],
  candlesticks: [],
  rsi: 0,
  macd: { macd: 0, signal: 0, histogram: 0 },
  bollinger: { upper: 0, middle: 0, lower: 0 },
  trendLines: {
    support: [],
    resistance: [],
    trend: 'neutral'
  },
  forecast: {
    shortTerm: 0,
    mediumTerm: 0,
    confidence: 0,
    direction: 'neutral'
  }
};

function calculateRSI(prices, period = 14) {
  if (prices.length < period + 1) return 50;

  let gains = 0;
  let losses = 0;

  for (let i = 1; i <= period; i++) {
    const change = prices[i] - prices[i - 1];
    if (change > 0) gains += change;
    else losses -= change;
  }

  const avgGain = gains / period;
  const avgLoss = losses / period;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

function calculateMACD(prices, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
  if (prices.length < slowPeriod) return { macd: 0, signal: 0, histogram: 0 };

  const ema12 = calculateEMA(prices, fastPeriod);
  const ema26 = calculateEMA(prices, slowPeriod);
  const macdLine = ema12 - ema26;

  return { macd: macdLine, signal: 0, histogram: macdLine };
}

function calculateEMA(prices, period) {
  if (prices.length === 0) return 0;
  const multiplier = 2 / (period + 1);
  let ema = prices[0];

  for (let i = 1; i < prices.length; i++) {
    ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
  }
  return ema;
}

function calculateBollingerBands(prices, period = 20, stdDev = 2) {
  if (prices.length < period) return { upper: 0, middle: 0, lower: 0 };

  const recentPrices = prices.slice(-period);
  const sma = recentPrices.reduce((sum, price) => sum + price, 0) / period;

  const variance = recentPrices.reduce((sum, price) => sum + Math.pow(price - sma, 2), 0) / period;
  const standardDeviation = Math.sqrt(variance);

  return {
    upper: sma + (standardDeviation * stdDev),
    middle: sma,
    lower: sma - (standardDeviation * stdDev)
  };
}

function calculateTrendLines(candlesticks) {
  if (candlesticks.length < 10) return { support: [], resistance: [], trend: 'neutral' };

  const highs = candlesticks.map(c => c.high);
  const lows = candlesticks.map(c => c.low);
  const closes = candlesticks.map(c => c.close);

  // Find support and resistance levels
  const support = findSupportResistance(lows, 'support');
  const resistance = findSupportResistance(highs, 'resistance');

  // Determine overall trend
  const trend = determineTrend(closes);

  return { support, resistance, trend };
}

function findSupportResistance(prices, type) {
  const levels = [];
  const window = 3; // Reduced window size for more sensitivity

  for (let i = window; i < prices.length - window; i++) {
    const current = prices[i];
    let isLevel = true;

    // Check if current price is a local min/max
    for (let j = i - window; j <= i + window; j++) {
      if (j === i) continue;
      if (type === 'support' && prices[j] < current) {
        isLevel = false;
        break;
      }
      if (type === 'resistance' && prices[j] > current) {
        isLevel = false;
        break;
      }
    }

    if (isLevel) {
      levels.push({
        price: current,
        time: i,
        strength: calculateLevelStrength(prices, current, i)
      });
    }
  }

  // Filter out levels that are too close to each other
  const filteredLevels = [];
  levels.sort((a, b) => b.strength - a.strength);

  for (const level of levels) {
    const tooClose = filteredLevels.some(existing =>
      Math.abs(existing.price - level.price) / level.price < 0.005 // 0.5% tolerance
    );
    if (!tooClose) {
      filteredLevels.push(level);
    }
  }

  // Return top 3 levels by strength
  return filteredLevels.slice(0, 3);
}

function calculateLevelStrength(prices, level, index) {
  let touches = 0;
  const tolerance = level * 0.002; // 0.2% tolerance

  for (let i = 0; i < prices.length; i++) {
    if (Math.abs(prices[i] - level) <= tolerance) {
      touches++;
    }
  }

  return touches;
}

function determineTrend(prices) {
  if (prices.length < 10) return 'neutral';

  const recent = prices.slice(-10);
  const older = prices.slice(-20, -10);

  const recentAvg = recent.reduce((sum, p) => sum + p, 0) / recent.length;
  const olderAvg = older.reduce((sum, p) => sum + p, 0) / older.length;

  const change = (recentAvg - olderAvg) / olderAvg;

  if (change > 0.01) return 'bullish';
  if (change < -0.01) return 'bearish';
  return 'neutral';
}

function generateForecast(candlesticks, indicators) {
  if (candlesticks.length < 20) return {
    shortTerm: 0,
    mediumTerm: 0,
    confidence: 0,
    direction: 'neutral'
  };

  const prices = candlesticks.map(c => c.close);
  const currentPrice = prices[prices.length - 1];

  // Simple linear regression for trend
  const n = prices.length;
  const x = Array.from({length: n}, (_, i) => i);
  const y = prices;

  const sumX = x.reduce((sum, val) => sum + val, 0);
  const sumY = y.reduce((sum, val) => sum + val, 0);
  const sumXY = x.reduce((sum, val, i) => sum + val * y[i], 0);
  const sumXX = x.reduce((sum, val) => sum + val * val, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  // Forecast future prices
  const shortTermForecast = slope * (n + 5) + intercept; // 5 periods ahead
  const mediumTermForecast = slope * (n + 15) + intercept; // 15 periods ahead

  // Calculate confidence based on R-squared
  const yMean = sumY / n;
  const ssRes = y.reduce((sum, val, i) => sum + Math.pow(val - (slope * i + intercept), 2), 0);
  const ssTot = y.reduce((sum, val) => sum + Math.pow(val - yMean, 2), 0);
  const rSquared = 1 - (ssRes / ssTot);
  const confidence = Math.max(0, Math.min(100, rSquared * 100));

  // Determine direction
  const direction = slope > 0 ? 'bullish' : slope < 0 ? 'bearish' : 'neutral';

  // Adjust forecast based on technical indicators
  let adjustment = 1;
  if (indicators.rsi > 70) adjustment *= 0.98; // Overbought
  if (indicators.rsi < 30) adjustment *= 1.02; // Oversold
  if (indicators.macd.macd > indicators.macd.signal) adjustment *= 1.01; // Bullish MACD

  return {
    shortTerm: shortTermForecast * adjustment,
    mediumTerm: mediumTermForecast * adjustment,
    confidence: confidence,
    direction: direction
  };
}

function generateSignals(data) {
  const signals = [];
  const currentPrice = parseFloat(data.price);
  const rsi = data.rsi;
  const macd = data.macd;
  const bollinger = data.bollinger;
  const forecast = data.forecast;

  if (rsi < 30) {
    signals.push({
      type: 'BUY',
      indicator: 'RSI',
      strength: 'STRONG',
      message: `RSI oversold at ${rsi.toFixed(2)}`,
      timestamp: new Date()
    });
  } else if (rsi > 70) {
    signals.push({
      type: 'SELL',
      indicator: 'RSI',
      strength: 'STRONG',
      message: `RSI overbought at ${rsi.toFixed(2)}`,
      timestamp: new Date()
    });
  }

  if (macd.macd > macd.signal && macd.histogram > 0) {
    signals.push({
      type: 'BUY',
      indicator: 'MACD',
      strength: 'MEDIUM',
      message: 'MACD bullish crossover',
      timestamp: new Date()
    });
  } else if (macd.macd < macd.signal && macd.histogram < 0) {
    signals.push({
      type: 'SELL',
      indicator: 'MACD',
      strength: 'MEDIUM',
      message: 'MACD bearish crossover',
      timestamp: new Date()
    });
  }

  if (currentPrice <= bollinger.lower) {
    signals.push({
      type: 'BUY',
      indicator: 'Bollinger Bands',
      strength: 'MEDIUM',
      message: 'Price touching lower Bollinger Band',
      timestamp: new Date()
    });
  } else if (currentPrice >= bollinger.upper) {
    signals.push({
      type: 'SELL',
      indicator: 'Bollinger Bands',
      strength: 'MEDIUM',
      message: 'Price touching upper Bollinger Band',
      timestamp: new Date()
    });
  }

  // Add forecast-based signals
  if (forecast.confidence > 60) {
    if (forecast.direction === 'bullish' && forecast.shortTerm > currentPrice * 1.005) {
      signals.push({
        type: 'BUY',
        indicator: 'Forecast',
        strength: 'MEDIUM',
        message: `Bullish forecast: ${((forecast.shortTerm - currentPrice) / currentPrice * 100).toFixed(2)}% potential upside`,
        timestamp: new Date()
      });
    } else if (forecast.direction === 'bearish' && forecast.shortTerm < currentPrice * 0.995) {
      signals.push({
        type: 'SELL',
        indicator: 'Forecast',
        strength: 'MEDIUM',
        message: `Bearish forecast: ${((currentPrice - forecast.shortTerm) / currentPrice * 100).toFixed(2)}% potential downside`,
        timestamp: new Date()
      });
    }
  }

  return signals;
}

async function initializeBinanceData() {
  try {
    const ticker = await binance.prices();
    const stats = await binance.prevDay('BTCUSDT');

    btcData.price = parseFloat(ticker.BTCUSDT);
    btcData.change24h = parseFloat(stats.priceChangePercent) || 0;
    btcData.volume = parseFloat(stats.volume) || 0;

    console.log('✅ Initial BTC data loaded:', btcData.price, 'Change:', btcData.change24h + '%');
  } catch (error) {
    console.error('❌ Error loading initial data:', error.message);
    btcData.change24h = 0;
    btcData.volume = 0;
  }
}

async function getCandlestickData() {
  try {
    const candles = await binance.candlesticks('BTCUSDT', '1m', false, { limit: 100 });
    const prices = candles.map(candle => parseFloat(candle[4])); // close prices

    btcData.candlesticks = candles.slice(-20).map(candle => ({
      time: candle[0],
      open: parseFloat(candle[1]),
      high: parseFloat(candle[2]),
      low: parseFloat(candle[3]),
      close: parseFloat(candle[4]),
      volume: parseFloat(candle[5])
    }));

    btcData.rsi = calculateRSI(prices);
    btcData.macd = calculateMACD(prices);
    btcData.bollinger = calculateBollingerBands(prices);

    // Calculate trend lines and forecast
    btcData.trendLines = calculateTrendLines(btcData.candlesticks);
    btcData.forecast = generateForecast(btcData.candlesticks, {
      rsi: btcData.rsi,
      macd: btcData.macd,
      bollinger: btcData.bollinger
    });

    console.log('📊 Trend Lines:', JSON.stringify(btcData.trendLines, null, 2));
    console.log('🔮 Forecast:', JSON.stringify(btcData.forecast, null, 2));

    btcData.signals = generateSignals(btcData);

  } catch (error) {
    console.error('❌ Error fetching candlestick data:', error.message);
  }
}

binance.websockets.miniTicker(markets => {
  if (markets && typeof markets === 'object') {
    const btcTicker = markets['BTCUSDT'];
    if (btcTicker) {
      btcData.price = parseFloat(btcTicker.close);
      btcData.change24h = parseFloat(btcTicker.percentChange);
      btcData.volume = parseFloat(btcTicker.volume);
    }
  }
});

setInterval(getCandlestickData, 60000);

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/api/btc-data', (req, res) => {
  res.json(btcData);
});

app.listen(port, async () => {
  console.log(`🚀 Bitcoin Signals Dashboard running on http://localhost:${port}`);
  await initializeBinanceData();
  await getCandlestickData();
});