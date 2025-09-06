#!/usr/bin/env python3
"""
Binance API Method Tester
Test different Binance API methods to find the correct ones
"""

import os
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Load environment variables
load_dotenv()


def test_binance_methods():
    """Test various Binance API methods to find correct ones"""

    print("🔍 Testing Binance API Methods...")
    print("=" * 60)

    # Initialize client
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'

    if api_key and api_secret:
        client = Client(api_key, api_secret, testnet=testnet)
        print(f"✅ Client initialized with API credentials (testnet={testnet})")
    else:
        client = Client()
        print("⚠️  Client initialized in public mode")

    symbol = 'BTCUSDT'
    print(f"🪙 Testing with symbol: {symbol}")
    print("-" * 60)

    # Test 1: Current Price
    print("1️⃣  Testing get_symbol_ticker...")
    try:
        price_data = client.get_symbol_ticker(symbol=symbol)
        print(f"   ✅ get_symbol_ticker: ${float(price_data['price']):,.2f}")
    except Exception as e:
        print(f"   ❌ get_symbol_ticker failed: {e}")

    # Test 2: 24hr Statistics - Method 1
    print("\n2️⃣  Testing get_ticker (24hr stats)...")
    try:
        ticker_data = client.get_ticker(symbol=symbol)
        print(f"   ✅ get_ticker works!")
        print(f"   📊 Price: ${float(ticker_data['lastPrice']):,.2f}")
        print(f"   📈 24h Change: {float(ticker_data['priceChangePercent']):.2f}%")
        print(f"   📊 24h Volume: {float(ticker_data['volume']):,.0f}")
        print(f"   📊 24h High: ${float(ticker_data['highPrice']):,.2f}")
        print(f"   📊 24h Low: ${float(ticker_data['lowPrice']):,.2f}")
    except Exception as e:
        print(f"   ❌ get_ticker failed: {e}")

    # Test 3: 24hr Statistics - Method 2
    print("\n3️⃣  Testing get_24hr_ticker...")
    try:
        ticker_24hr = client.get_24hr_ticker(symbol=symbol)
        print(f"   ✅ get_24hr_ticker works!")
        print(f"   📊 24h Change: {float(ticker_24hr['priceChangePercent']):.2f}%")
    except AttributeError as e:
        print(f"   ❌ get_24hr_ticker method doesn't exist: {e}")
    except Exception as e:
        print(f"   ❌ get_24hr_ticker failed: {e}")

    # Test 4: All available methods
    print("\n4️⃣  Available ticker-related methods:")
    ticker_methods = [method for method in dir(client) if 'ticker' in method.lower()]
    for method in ticker_methods:
        print(f"   • {method}")

    # Test 5: Book Ticker (bid/ask)
    print("\n5️⃣  Testing get_orderbook_ticker...")
    try:
        book_ticker = client.get_orderbook_ticker(symbol=symbol)
        print(f"   ✅ get_orderbook_ticker works!")
        print(f"   💰 Bid: ${float(book_ticker['bidPrice']):,.2f}")
        print(f"   💰 Ask: ${float(book_ticker['askPrice']):,.2f}")
    except Exception as e:
        print(f"   ❌ get_orderbook_ticker failed: {e}")

    # Test 6: All Tickers
    print("\n6️⃣  Testing get_ticker (all symbols)...")
    try:
        all_tickers = client.get_ticker()
        print(f"   ✅ get_ticker (all) works! Got {len(all_tickers)} symbols")

        # Find USDT pairs
        usdt_pairs = [t for t in all_tickers if t['symbol'].endswith('USDT')][:5]
        print(f"   📊 Top 5 USDT pairs by volume:")
        for ticker in sorted(usdt_pairs, key=lambda x: float(x['volume']), reverse=True)[:5]:
            print(f"      • {ticker['symbol']}: Vol {float(ticker['volume']):,.0f}")

    except Exception as e:
        print(f"   ❌ get_ticker (all) failed: {e}")

    print("\n" + "=" * 60)
    print("🎯 RECOMMENDED METHOD MAPPING:")
    print("=" * 60)
    print("✅ Current Price: client.get_symbol_ticker(symbol='BTCUSDT')")
    print("✅ 24hr Stats: client.get_ticker(symbol='BTCUSDT')")
    print("✅ Bid/Ask: client.get_orderbook_ticker(symbol='BTCUSDT')")
    print("✅ All Tickers: client.get_ticker()")
    print("\n❌ AVOID: get_24hr_ticker() - This method doesn't exist!")


def test_corrected_market_data():
    """Test the corrected market data function"""
    print("\n" + "🧪 Testing Corrected Market Data Function...")
    print("=" * 60)

    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'

    if api_key and api_secret:
        client = Client(api_key, api_secret, testnet=testnet)
    else:
        client = Client()

    def get_market_data_corrected(symbol: str):
        """Corrected market data function"""
        try:
            # Get current price
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            # Get 24h statistics - CORRECTED METHOD
            stats = client.get_ticker(symbol=symbol)

            return {
                'symbol': symbol,
                'price': current_price,
                'volume': float(stats['volume']),
                'change_24h': float(stats['priceChangePercent']),
                'high_24h': float(stats['highPrice']),
                'low_24h': float(stats['lowPrice']),
                'open_price': float(stats['openPrice']),
                'close_price': float(stats['prevClosePrice'])
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    # Test the corrected function
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

    for symbol in symbols:
        print(f"\n🪙 Testing {symbol}:")
        data = get_market_data_corrected(symbol)
        if data:
            print(f"   ✅ Price: ${data['price']:,.2f}")
            print(f"   📈 24h Change: {data['change_24h']:+.2f}%")
            print(f"   📊 24h Volume: {data['volume']:,.0f}")
            print(f"   📊 24h High: ${data['high_24h']:,.2f}")
            print(f"   📊 24h Low: ${data['low_24h']:,.2f}")
        else:
            print(f"   ❌ Failed to get data for {symbol}")


if __name__ == "__main__":
    test_binance_methods()
    test_corrected_market_data()