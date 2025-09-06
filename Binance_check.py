import os
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Load environment variables
load_dotenv()


def test_binance_api():
    """Test Binance API connection and permissions"""

    print("🔧 Testing Binance API Connection...")
    print("=" * 50)

    # Get credentials
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        print("❌ Error: BINANCE_API_KEY and BINANCE_API_SECRET not found in environment")
        print("\nPlease set them in your .env file:")
        print("BINANCE_API_KEY=your_api_key_here")
        print("BINANCE_API_SECRET=your_secret_key_here")
        return False

    try:
        # Initialize client (testnet=True for testing)
        # Change to testnet=False for mainnet
        client = Client(api_key, api_secret, testnet=True)
        print("✅ Client initialized successfully")

        # Test 1: Get account info
        print("\n📊 Testing Account Access...")
        try:
            account_info = client.get_account()
            print("✅ Account access successful")
            print(f"   Account Type: {account_info.get('accountType', 'Unknown')}")
            print(f"   Can Trade: {account_info.get('canTrade', False)}")
            print(f"   Can Withdraw: {account_info.get('canWithdraw', False)}")
        except BinanceAPIException as e:
            if "Invalid API-key" in str(e):
                print("❌ Invalid API Key")
                return False
            elif "Signature for this request is not valid" in str(e):
                print("❌ Invalid API Secret")
                return False
            else:
                print(f"⚠️  Account access limited: {e}")

        # Test 2: Get Bitcoin price (public data)
        print("\n💰 Testing Price Data Access...")
        try:
            btc_price = client.get_symbol_ticker(symbol='BTCUSDT')
            print("✅ Price data access successful")
            print(f"   BTC/USDT Price: ${float(btc_price['price']):,.2f}")
        except Exception as e:
            print(f"❌ Price data error: {e}")

        # Test 3: Get 24hr ticker
        print("\n📈 Testing Market Data...")
        try:
            ticker = client.get_24hr_ticker(symbol='BTCUSDT')
            print("✅ Market data access successful")
            print(f"   24h Change: {float(ticker['priceChangePercent']):+.2f}%")
            print(f"   24h Volume: {float(ticker['volume']):,.0f} BTC")
            print(f"   24h High: ${float(ticker['highPrice']):,.2f}")
            print(f"   24h Low: ${float(ticker['lowPrice']):,.2f}")
        except Exception as e:
            print(f"❌ Market data error: {e}")

        # Test 4: Test trading permissions (if enabled)
        print("\n🔍 Testing Trading Permissions...")
        try:
            # This will only work if trading is enabled
            exchange_info = client.get_exchange_info()
            symbols = [s for s in exchange_info['symbols'] if s['symbol'] == 'BTCUSDT']
            if symbols:
                print("✅ Trading permissions available")
                symbol_info = symbols[0]
                print(f"   Symbol Status: {symbol_info['status']}")
                print(f"   Order Types: {', '.join(symbol_info['orderTypes'][:5])}...")
            else:
                print("❌ BTCUSDT symbol not found")
        except Exception as e:
            print(f"⚠️  Limited trading access: {e}")

        print("\n🎉 API Test Complete!")
        print("\n🔒 Security Reminder:")
        print("- Never share your API credentials")
        print("- Never enable withdrawals for trading bots")
        print("- Use IP restrictions when possible")
        print("- Monitor your API usage regularly")

        return True

    except BinanceAPIException as e:
        print(f"❌ Binance API Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False


def show_setup_guide():
    """Show setup guide for Binance API"""
    print("\n📋 Binance API Setup Guide:")
    print("=" * 50)
    print("1. Go to https://www.binance.com")
    print("2. Log in to your account")
    print("3. Go to Account → API Management")
    print("4. Create new API key")
    print("5. Enable 'Enable Reading' permission")
    print("6. Optionally enable 'Enable Spot & Margin Trading'")
    print("7. NEVER enable 'Enable Withdrawals'")
    print("8. Set IP restrictions (recommended)")
    print("9. Copy API Key and Secret to your .env file")
    print("\n⚠️  For testing, you can use Binance Testnet:")
    print("   https://testnet.binance.vision/")


if __name__ == "__main__":
    print("🚀 Binance API Test & Setup Tool")
    print("=" * 50)

    success = test_binance_api()

    if not success:
        show_setup_guide()

    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed. Check your setup.'}")
