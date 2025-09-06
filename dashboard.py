import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import threading
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import requests
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bitcoin_dashboard.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence: Optional[str] = None
    timestamp: datetime = None
    source: str = "Manual"
    binance_price: Optional[float] = None
    price_difference: Optional[float] = None
    volume_24h: Optional[float] = None
    change_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    status: str = "ACTIVE"  # ACTIVE, EXECUTED, CANCELLED
    pnl: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'price': self.price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'binance_price': self.binance_price,
            'price_difference': self.price_difference,
            'volume_24h': self.volume_24h,
            'change_24h': self.change_24h,
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'status': self.status,
            'pnl': self.pnl
        }


class BinanceManager:
    """Enhanced Binance API manager"""

    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'

        if self.api_key and self.api_secret:
            self.client = BinanceClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            self.authenticated = True
            logger.info("✅ Binance client initialized with API credentials")
        else:
            self.client = BinanceClient()
            self.authenticated = False
            logger.info("⚠️ Binance client initialized in public mode")

        self.symbol_mapping = {
            'BTC': 'BTCUSDT',
            'ETH': 'ETHUSDT',
            'BNB': 'BNBUSDT',
            'ADA': 'ADAUSDT',
            'XRP': 'XRPUSDT',
            'SOL': 'SOLUSDT',
            'DOT': 'DOTUSDT',
            'MATIC': 'MATICUSDT'
        }

    def get_symbol(self, symbol: str) -> str:
        """Convert symbol to Binance format"""
        symbol = symbol.upper().replace('USDT', '')
        return self.symbol_mapping.get(symbol, symbol + 'USDT')

    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market data"""
        try:
            binance_symbol = self.get_symbol(symbol)

            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=binance_symbol)
            current_price = float(ticker['price'])

            # Get 24h statistics
            stats = self.client.get_24hr_ticker(symbol=binance_symbol)

            return {
                'symbol': binance_symbol,
                'price': current_price,
                'volume': float(stats['volume']),
                'change_24h': float(stats['priceChangePercent']),
                'high_24h': float(stats['highPrice']),
                'low_24h': float(stats['lowPrice']),
                'open_price': float(stats['openPrice']),
                'close_price': float(stats['prevClosePrice']),
                'bid': float(stats['bidPrice']) if 'bidPrice' in stats else current_price,
                'ask': float(stats['askPrice']) if 'askPrice' in stats else current_price
            }

        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information"""
        if not self.authenticated:
            return None

        try:
            account = self.client.get_account()
            balances = [
                {
                    'asset': balance['asset'],
                    'free': float(balance['free']),
                    'locked': float(balance['locked']),
                    'total': float(balance['free']) + float(balance['locked'])
                }
                for balance in account['balances']
                if float(balance['free']) > 0 or float(balance['locked']) > 0
            ]

            return {
                'account_type': account.get('accountType', 'Unknown'),
                'can_trade': account.get('canTrade', False),
                'can_withdraw': account.get('canWithdraw', False),
                'balances': balances,
                'total_balance_btc': sum(float(b['free']) + float(b['locked'])
                                         for b in account['balances'] if b['asset'] == 'BTC')
            }

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    def get_top_symbols(self) -> List[str]:
        """Get top trading symbols"""
        try:
            tickers = self.client.get_ticker()
            # Filter USDT pairs and sort by volume
            usdt_pairs = [
                ticker for ticker in tickers
                if ticker['symbol'].endswith('USDT') and ticker['symbol'] != 'USDT'
            ]

            # Sort by volume and get top 20
            sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['volume']), reverse=True)
            return [pair['symbol'] for pair in sorted_pairs[:20]]

        except Exception as e:
            logger.error(f"Error getting top symbols: {e}")
            return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']


class SignalAnalyzer:
    """Analyze and generate trading signals"""

    def __init__(self, binance_manager: BinanceManager):
        self.binance = binance_manager

    def analyze_rsi_signal(self, symbol: str, period: int = 14) -> Optional[TradingSignal]:
        """Generate RSI-based signal (simplified)"""
        try:
            market_data = self.binance.get_market_data(symbol)
            if not market_data:
                return None

            # Simplified RSI calculation based on price change
            change_24h = market_data['change_24h']
            current_price = market_data['price']

            # Basic signal logic
            if change_24h < -5:  # Oversold condition
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type='BUY',
                    price=current_price,
                    target_price=current_price * 1.05,  # 5% profit target
                    stop_loss=current_price * 0.97,  # 3% stop loss
                    confidence='HIGH' if change_24h < -10 else 'MEDIUM',
                    source='RSI Analysis',
                    binance_price=current_price,
                    volume_24h=market_data['volume'],
                    change_24h=change_24h,
                    high_24h=market_data['high_24h'],
                    low_24h=market_data['low_24h']
                )

            elif change_24h > 5:  # Overbought condition
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type='SELL',
                    price=current_price,
                    target_price=current_price * 0.95,  # 5% profit target
                    stop_loss=current_price * 1.03,  # 3% stop loss
                    confidence='HIGH' if change_24h > 10 else 'MEDIUM',
                    source='RSI Analysis',
                    binance_price=current_price,
                    volume_24h=market_data['volume'],
                    change_24h=change_24h,
                    high_24h=market_data['high_24h'],
                    low_24h=market_data['low_24h']
                )
            else:
                return None

            return signal

        except Exception as e:
            logger.error(f"Error analyzing RSI signal: {e}")
            return None


class DashboardApp:
    """Main dashboard application"""

    def __init__(self):
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.secret_key = os.getenv('SECRET_KEY', 'bitcoin-signals-dashboard-key')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        self.binance = BinanceManager()
        self.analyzer = SignalAnalyzer(self.binance)
        self.signals_history: List[TradingSignal] = []
        self.watchlist: List[str] = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT']
        self.running = True

        self.setup_routes()
        self.setup_socketio()

        # Start background tasks
        self.start_background_tasks()

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/api/signals')
        def get_signals():
            """Get all signals"""
            return jsonify([signal.to_dict() for signal in self.signals_history[-50:]])

        @self.app.route('/api/signals', methods=['POST'])
        def add_signal():
            """Add manual signal"""
            try:
                data = request.json

                # Validate required fields
                required_fields = ['symbol', 'signal_type']
                if not all(field in data for field in required_fields):
                    return jsonify({'error': 'Missing required fields'}), 400

                # Get market data
                market_data = self.binance.get_market_data(data['symbol'])

                signal = TradingSignal(
                    symbol=self.binance.get_symbol(data['symbol']),
                    signal_type=data['signal_type'].upper(),
                    price=data.get('price', market_data['price'] if market_data else None),
                    target_price=data.get('target_price'),
                    stop_loss=data.get('stop_loss'),
                    confidence=data.get('confidence', 'MEDIUM'),
                    source='Manual Entry',
                    binance_price=market_data['price'] if market_data else None,
                    volume_24h=market_data['volume'] if market_data else None,
                    change_24h=market_data['change_24h'] if market_data else None,
                    high_24h=market_data['high_24h'] if market_data else None,
                    low_24h=market_data['low_24h'] if market_data else None
                )

                self.signals_history.append(signal)

                # Emit to all clients
                self.socketio.emit('new_signal', signal.to_dict())

                return jsonify({'success': True, 'signal': signal.to_dict()})

            except Exception as e:
                logger.error(f"Error adding signal: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/market/<symbol>')
        def get_market_data(symbol):
            """Get market data for a symbol"""
            data = self.binance.get_market_data(symbol)
            if data:
                return jsonify(data)
            return jsonify({'error': 'Symbol not found'}), 404

        @self.app.route('/api/account')
        def get_account():
            """Get account information"""
            account_info = self.binance.get_account_info()
            if account_info:
                return jsonify(account_info)
            return jsonify({'error': 'Account info not available'}), 404

        @self.app.route('/api/watchlist')
        def get_watchlist():
            """Get watchlist with market data"""
            watchlist_data = []
            for symbol in self.watchlist:
                market_data = self.binance.get_market_data(symbol)
                if market_data:
                    watchlist_data.append(market_data)
            return jsonify(watchlist_data)

        @self.app.route('/api/watchlist', methods=['POST'])
        def add_to_watchlist():
            """Add symbol to watchlist"""
            data = request.json
            symbol = self.binance.get_symbol(data.get('symbol', ''))

            if symbol and symbol not in self.watchlist:
                self.watchlist.append(symbol)
                return jsonify({'success': True})
            return jsonify({'error': 'Invalid symbol or already in watchlist'}), 400

        @self.app.route('/api/top-symbols')
        def get_top_symbols():
            """Get top trading symbols"""
            symbols = self.binance.get_top_symbols()
            return jsonify(symbols)

    def setup_socketio(self):
        """Setup SocketIO events"""

        @self.socketio.on('connect')
        def handle_connect():
            emit('connected', {'status': 'Connected to Bitcoin Signals Dashboard'})

        @self.socketio.on('request_market_update')
        def handle_market_update(data):
            """Send real-time market update"""
            symbol = data.get('symbol')
            if symbol:
                market_data = self.binance.get_market_data(symbol)
                if market_data:
                    emit('market_update', market_data)

    def start_background_tasks(self):
        """Start background monitoring tasks"""

        def market_monitor():
            """Monitor market and generate automatic signals"""
            while self.running:
                try:
                    for symbol in self.watchlist:
                        # Analyze for potential signals
                        signal = self.analyzer.analyze_rsi_signal(symbol)
                        if signal:
                            # Check if we already have a recent signal for this symbol
                            recent_signals = [
                                s for s in self.signals_history[-10:]
                                if s.symbol == signal.symbol and
                                   (datetime.now() - s.timestamp).seconds < 3600  # 1 hour
                            ]

                            if not recent_signals:
                                logger.info(f"🚨 Auto Signal: {signal.signal_type} {signal.symbol}")
                                self.signals_history.append(signal)
                                self.socketio.emit('new_signal', signal.to_dict())

                    # Send market updates
                    for symbol in self.watchlist:
                        market_data = self.binance.get_market_data(symbol)
                        if market_data:
                            self.socketio.emit('market_update', market_data)

                    time.sleep(30)  # Check every 30 seconds

                except Exception as e:
                    logger.error(f"Error in market monitor: {e}")
                    time.sleep(60)

        # Start market monitor in background thread
        monitor_thread = threading.Thread(target=market_monitor, daemon=True)
        monitor_thread.start()

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the dashboard application"""
        logger.info(f"🚀 Starting Bitcoin Signals Dashboard on http://{host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)


if __name__ == "__main__":
    # Railway and production configuration
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'

    logger.info(f"🚀 Starting Bitcoin Signals Dashboard on http://{host}:{port}")

    # Additional Railway-specific configurations
    if os.getenv('RAILWAY_ENVIRONMENT'):
        logger.info("🚂 Running on Railway")
        # Railway-specific settings
        dashboard = DashboardApp()
        dashboard.socketio.run(
            dashboard.app,
            host=host,
            port=port,
            debug=debug,
            allow_unsafe_werkzeug=True  # Required for Railway WebSocket support
        )
    else:
        # Local development
        dashboard = DashboardApp()
        dashboard.run(host=host, port=port, debug=debug)