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
import random

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
            try:
                self.client = BinanceClient(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.testnet
                )
                self.authenticated = True
                logger.info("✅ Binance client initialized with API credentials")
            except Exception as e:
                logger.error(f"❌ Error initializing Binance client: {e}")
                self.client = BinanceClient()
                self.authenticated = False
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
            'MATIC': 'MATICUSDT',
            'AVAX': 'AVAXUSDT',
            'LINK': 'LINKUSDT'
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
            # Get all tickers - correct method name
            tickers = self.client.get_ticker()
            # Filter USDT pairs and sort by volume
            usdt_pairs = [
                ticker for ticker in tickers
                if ticker['symbol'].endswith('USDT') and ticker['symbol'] != 'USDTUSDT'
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
        """Generate RSI-based signal (enhanced with more realistic conditions)"""
        try:
            market_data = self.binance.get_market_data(symbol)
            if not market_data:
                return None

            # Get price change data
            change_24h = market_data['change_24h']
            current_price = market_data['price']
            volume_24h = market_data['volume']

            # Enhanced signal logic with more realistic conditions
            signal = None

            # Strong oversold condition (more likely to trigger)
            if change_24h < -3:  # Lowered from -5% to -3%
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type='BUY',
                    price=current_price,
                    target_price=current_price * 1.05,  # 5% profit target
                    stop_loss=current_price * 0.97,  # 3% stop loss
                    confidence='HIGH' if change_24h < -7 else 'MEDIUM',
                    source='RSI Analysis',
                    binance_price=current_price,
                    volume_24h=volume_24h,
                    change_24h=change_24h,
                    high_24h=market_data['high_24h'],
                    low_24h=market_data['low_24h']
                )

            # Strong overbought condition
            elif change_24h > 3:  # Lowered from +5% to +3%
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type='SELL',
                    price=current_price,
                    target_price=current_price * 0.95,  # 5% profit target
                    stop_loss=current_price * 1.03,  # 3% stop loss
                    confidence='HIGH' if change_24h > 7 else 'MEDIUM',
                    source='RSI Analysis',
                    binance_price=current_price,
                    volume_24h=volume_24h,
                    change_24h=change_24h,
                    high_24h=market_data['high_24h'],
                    low_24h=market_data['low_24h']
                )

            # Volume-based signals (new)
            elif volume_24h > 0:  # If we have volume data
                # Random signal generation for demo (remove in production)
                if random.random() < 0.1:  # 10% chance every check
                    signal_type = random.choice(['BUY', 'SELL'])
                    confidence = random.choice(['HIGH', 'MEDIUM', 'LOW'])

                    if signal_type == 'BUY':
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=signal_type,
                            price=current_price,
                            target_price=current_price * random.uniform(1.02, 1.08),
                            stop_loss=current_price * random.uniform(0.95, 0.98),
                            confidence=confidence,
                            source='Volume Analysis',
                            binance_price=current_price,
                            volume_24h=volume_24h,
                            change_24h=change_24h,
                            high_24h=market_data['high_24h'],
                            low_24h=market_data['low_24h']
                        )
                    else:
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=signal_type,
                            price=current_price,
                            target_price=current_price * random.uniform(0.92, 0.98),
                            stop_loss=current_price * random.uniform(1.02, 1.05),
                            confidence=confidence,
                            source='Volume Analysis',
                            binance_price=current_price,
                            volume_24h=volume_24h,
                            change_24h=change_24h,
                            high_24h=market_data['high_24h'],
                            low_24h=market_data['low_24h']
                        )

            return signal

        except Exception as e:
            logger.error(f"Error analyzing signal for {symbol}: {e}")
            return None

    def generate_demo_signals(self, symbols: List[str]) -> List[TradingSignal]:
        """Generate some demo signals for testing"""
        demo_signals = []

        for symbol in symbols[:3]:  # Generate for first 3 symbols
            try:
                market_data = self.binance.get_market_data(symbol)
                if not market_data:
                    continue

                # Create demo signal
                signal_type = random.choice(['BUY', 'SELL'])
                confidence = random.choice(['HIGH', 'MEDIUM'])
                current_price = market_data['price']

                if signal_type == 'BUY':
                    target_price = current_price * random.uniform(1.03, 1.07)
                    stop_loss = current_price * random.uniform(0.96, 0.98)
                else:
                    target_price = current_price * random.uniform(0.93, 0.97)
                    stop_loss = current_price * random.uniform(1.02, 1.04)

                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    confidence=confidence,
                    source='Demo Signal',
                    binance_price=current_price,
                    volume_24h=market_data['volume'],
                    change_24h=market_data['change_24h'],
                    high_24h=market_data['high_24h'],
                    low_24h=market_data['low_24h'],
                    timestamp=datetime.now() - timedelta(minutes=random.randint(1, 60))
                )

                demo_signals.append(signal)

            except Exception as e:
                logger.error(f"Error generating demo signal for {symbol}: {e}")
                continue

        return demo_signals


class DashboardApp:
    """Main dashboard application"""

    def __init__(self):
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.secret_key = os.getenv('SECRET_KEY', 'bitcoin-signals-dashboard-key')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        self.binance = BinanceManager()
        self.analyzer = SignalAnalyzer(self.binance)
        self.signals_history: List[TradingSignal] = []
        self.watchlist: List[str] = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']
        self.running = True

        self.setup_routes()
        self.setup_socketio()

        # Generate initial demo signals
        self.generate_initial_signals()

        # Start background tasks
        self.start_background_tasks()

    def generate_initial_signals(self):
        """Generate some initial signals for demonstration"""
        try:
            logger.info("🎯 Generating initial demo signals...")
            demo_signals = self.analyzer.generate_demo_signals(self.watchlist)
            self.signals_history.extend(demo_signals)
            logger.info(f"✅ Generated {len(demo_signals)} initial signals")
        except Exception as e:
            logger.error(f"❌ Error generating initial signals: {e}")

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/api/signals')
        def get_signals():
            """Get all signals"""
            try:
                signals_data = [signal.to_dict() for signal in self.signals_history[-50:]]
                logger.info(f"📡 Returning {len(signals_data)} signals to client")
                return jsonify(signals_data)
            except Exception as e:
                logger.error(f"❌ Error getting signals: {e}")
                return jsonify([])

        @self.app.route('/api/signals', methods=['POST'])
        def add_signal():
            """Add manual signal"""
            try:
                data = request.json
                logger.info(f"📝 Adding manual signal: {data}")

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
                logger.info(f"✅ Manual signal added: {signal.signal_type} {signal.symbol}")

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
            logger.info(f"📊 Returning {len(watchlist_data)} watchlist items")
            return jsonify(watchlist_data)

        @self.app.route('/api/watchlist', methods=['POST'])
        def add_to_watchlist():
            """Add symbol to watchlist"""
            data = request.json
            symbol = self.binance.get_symbol(data.get('symbol', ''))

            if symbol and symbol not in self.watchlist:
                self.watchlist.append(symbol)
                logger.info(f"✅ Added {symbol} to watchlist")
                return jsonify({'success': True})
            return jsonify({'error': 'Invalid symbol or already in watchlist'}), 400

        @self.app.route('/api/top-symbols')
        def get_top_symbols():
            """Get top trading symbols"""
            symbols = self.binance.get_top_symbols()
            return jsonify(symbols)

        # Add endpoint to generate demo signal manually
        @self.app.route('/api/generate-demo-signal', methods=['POST'])
        def generate_demo_signal():
            """Generate a demo signal for testing"""
            try:
                symbol = random.choice(self.watchlist)
                demo_signals = self.analyzer.generate_demo_signals([symbol])

                if demo_signals:
                    signal = demo_signals[0]
                    self.signals_history.append(signal)
                    logger.info(f"🎯 Generated demo signal: {signal.signal_type} {signal.symbol}")

                    # Emit to all clients
                    self.socketio.emit('new_signal', signal.to_dict())

                    return jsonify({'success': True, 'signal': signal.to_dict()})
                else:
                    return jsonify({'error': 'Failed to generate demo signal'}), 500

            except Exception as e:
                logger.error(f"Error generating demo signal: {e}")
                return jsonify({'error': str(e)}), 500

    def setup_socketio(self):
        """Setup SocketIO events"""

        @self.socketio.on('connect')
        def handle_connect():
            logger.info("🔗 Client connected via WebSocket")
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
            logger.info("🚀 Starting market monitor background task")

            while self.running:
                try:
                    logger.info("🔍 Checking for new signals...")
                    signals_generated = 0

                    for symbol in self.watchlist:
                        # Analyze for potential signals
                        signal = self.analyzer.analyze_rsi_signal(symbol)
                        if signal:
                            # Check if we already have a recent signal for this symbol
                            recent_signals = [
                                s for s in self.signals_history[-10:]
                                if s.symbol == signal.symbol and
                                   (datetime.now() - s.timestamp).seconds < 1800  # 30 minutes
                            ]

                            if not recent_signals:
                                logger.info(
                                    f"🚨 Auto Signal Generated: {signal.signal_type} {signal.symbol} at ${signal.price}")
                                self.signals_history.append(signal)
                                self.socketio.emit('new_signal', signal.to_dict())
                                signals_generated += 1

                    if signals_generated > 0:
                        logger.info(f"✅ Generated {signals_generated} new signals")

                    # Send market updates to connected clients
                    for symbol in self.watchlist:
                        market_data = self.binance.get_market_data(symbol)
                        if market_data:
                            self.socketio.emit('market_update', market_data)

                    logger.info("💤 Market monitor sleeping for 60 seconds...")
                    time.sleep(60)  # Check every 60 seconds

                except Exception as e:
                    logger.error(f"❌ Error in market monitor: {e}")
                    time.sleep(60)

        # Start market monitor in background thread
        monitor_thread = threading.Thread(target=market_monitor, daemon=True)
        monitor_thread.start()
        logger.info("✅ Background market monitor started")

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the dashboard application"""
        logger.info(f"🚀 Starting Bitcoin Signals Dashboard on http://{host}:{port}")
        logger.info(f"📊 Initial signals count: {len(self.signals_history)}")
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
        logger.info("💻 Running locally")
        dashboard = DashboardApp()
        dashboard.run(host=host, port=port, debug=debug)