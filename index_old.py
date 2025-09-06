import asyncio
import re
import logging
import os
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from threading import Thread
import signal
import sys
import math

# Required packages: pip install telethon python-dotenv python-binance aiofiles
from telethon import TelegramClient, events
from telethon.tl.types import Channel
from telethon.errors import SessionPasswordNeededError
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException, BinanceOrderException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_flag = False


@dataclass
class TradingSignal:
    """Enhanced trading signal with Binance integration"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence: Optional[str] = None
    timestamp: datetime = None
    raw_message: str = ""
    channel_name: str = ""
    binance_price: Optional[float] = None
    price_difference: Optional[float] = None
    volume_24h: Optional[float] = None

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
            'raw_message': self.raw_message,
            'channel_name': self.channel_name,
            'binance_price': self.binance_price,
            'price_difference': self.price_difference,
            'volume_24h': self.volume_24h
        }


class BinanceIntegration:
    """Binance API integration for real-time price data and trading"""

    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret

        # Initialize Binance client
        if api_key and api_secret:
            self.client = BinanceClient(api_key, api_secret,
                                        testnet=os.getenv('BINANCE_TESTNET', 'true').lower() == 'true')
            logger.info("✅ Binance client initialized with API credentials")
        else:
            self.client = BinanceClient()  # Public API only
            logger.info("⚠️ Binance client initialized in public mode (no trading)")

        self.symbol_mapping = {
            'BTCUSD': 'BTCUSDT',
            'BTC': 'BTCUSDT',
            'BITCOIN': 'BTCUSDT'
        }

    def get_symbol(self, symbol: str) -> str:
        """Convert signal symbol to Binance format"""
        return self.symbol_mapping.get(symbol.upper(), symbol.upper() + 'USDT')

    async def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price and market data from Binance"""
        try:
            binance_symbol = self.get_symbol(symbol)

            # Get price ticker
            ticker = self.client.get_symbol_ticker(symbol=binance_symbol)
            current_price = float(ticker['price'])

            # Get 24h statistics
            stats = self.client.get_24hr_ticker(symbol=binance_symbol)

            return {
                'price': current_price,
                'volume': float(stats['volume']),
                'change_24h': float(stats['priceChangePercent']),
                'high_24h': float(stats['highPrice']),
                'low_24h': float(stats['lowPrice'])
            }

        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting price data: {e}")
            return None

    def calculate_position_size(self, balance_usdt: float, risk_percentage: float = 2.0,
                                entry_price: float = None, stop_loss: float = None) -> float:
        """Calculate position size based on risk management"""
        if not entry_price or not stop_loss:
            return 0

        risk_amount = balance_usdt * (risk_percentage / 100)
        price_difference = abs(entry_price - stop_loss)

        if price_difference == 0:
            return 0

        position_size = risk_amount / price_difference
        return round(position_size, 6)

    def place_test_order(self, symbol: str, side: str, quantity: float, price: float = None) -> Dict[str, Any]:
        """Place a test order (paper trading)"""
        try:
            binance_symbol = self.get_symbol(symbol)

            order_params = {
                'symbol': binance_symbol,
                'side': side.upper(),  # BUY or SELL
                'type': 'MARKET' if not price else 'LIMIT',
                'quantity': quantity,
            }

            if price:
                order_params['price'] = price
                order_params['timeInForce'] = 'GTC'

            # For demonstration - this would be a real order in production
            logger.info(f"📝 Test Order: {order_params}")

            return {
                'success': True,
                'orderId': f"test_{int(time.time())}",
                'symbol': binance_symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'status': 'FILLED'
            }

        except Exception as e:
            logger.error(f"Error placing test order: {e}")
            return {'success': False, 'error': str(e)}


class EnhancedSignalExtractor:
    """Enhanced signal extraction with better patterns"""

    def __init__(self):
        # Cryptocurrency patterns
        self.crypto_patterns = {
            'BTC': [r'BTC[/\s]*USD?T?', r'BITCOIN', r'₿', r'#BTC', r'\$BTC'],
            'ETH': [r'ETH[/\s]*USD?T?', r'ETHEREUM', r'#ETH', r'\$ETH'],
            'BNB': [r'BNB[/\s]*USD?T?', r'BINANCE\s*COIN', r'#BNB'],
        }

        # Signal patterns with emojis
        self.buy_patterns = [
            r'\b(BUY|LONG|BULL|BULLISH|ENTER\s*LONG|CALL)\b',
            r'🔥', r'🚀', r'📈', r'🟢', r'⬆️', r'💚', r'🎯'
        ]

        self.sell_patterns = [
            r'\b(SELL|SHORT|BEAR|BEARISH|ENTER\s*SHORT|PUT)\b',
            r'📉', r'🔴', r'⬇️', r'💥', r'❤️', r'⚠️'
        ]

        # Price extraction patterns
        self.price_patterns = [
            r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{1,4})?)',  # $45,000.50
            r'(\d+(?:\.\d+)?)[Kk]',  # 45.5K
            r'(\d+(?:\.\d+)?)[Mm]',  # 1.2M
        ]

        # Target and stop loss patterns
        self.target_patterns = [
            r'(?:TARGET|TP\s*\d*|TAKE\s*PROFIT).*?[\$]?(\d+(?:,\d{3})*(?:\.\d{1,4})?)',
            r'🎯.*?[\$]?(\d+(?:,\d{3})*(?:\.\d{1,4})?)',
            r'TARGET.*?(\d+(?:\.\d+)?)[Kk]'
        ]

        self.stop_loss_patterns = [
            r'(?:STOP\s*LOSS|SL|STOP).*?[\$]?(\d+(?:,\d{3})*(?:\.\d{1,4})?)',
            r'🛑.*?[\$]?(\d+(?:,\d{3})*(?:\.\d{1,4})?)',
            r'(?:INVALIDATION|INVALID).*?[\$]?(\d+(?:,\d{3})*(?:\.\d{1,4})?)'
        ]

    def extract_signal(self, message: str, channel_name: str = "") -> Optional[TradingSignal]:
        """Extract enhanced trading signal from message"""
        message_upper = message.upper()

        # Detect cryptocurrency
        detected_crypto = None
        for crypto, patterns in self.crypto_patterns.items():
            if any(re.search(pattern, message_upper, re.IGNORECASE) for pattern in patterns):
                detected_crypto = crypto
                break

        if not detected_crypto:
            return None

        # Determine signal type
        signal_type = None
        buy_score = sum(1 for pattern in self.buy_patterns
                        if re.search(pattern, message_upper, re.IGNORECASE))
        sell_score = sum(1 for pattern in self.sell_patterns
                         if re.search(pattern, message_upper, re.IGNORECASE))

        if buy_score > sell_score and buy_score > 0:
            signal_type = "BUY"
        elif sell_score > buy_score and sell_score > 0:
            signal_type = "SELL"

        if not signal_type:
            return None

        # Extract prices and targets
        price = self._extract_price(message)
        target_price = self._extract_target(message)
        stop_loss = self._extract_stop_loss(message)
        confidence = self._extract_confidence(message)

        return TradingSignal(
            symbol=f"{detected_crypto}USDT",
            signal_type=signal_type,
            price=price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            raw_message=message,
            channel_name=channel_name
        )

    def _extract_price(self, message: str) -> Optional[float]:
        """Enhanced price extraction"""
        # Look for price near common keywords
        price_contexts = [
            r'(?:PRICE|ENTRY|CURRENT|NOW|AT).*?[\$]?(\d+(?:,\d{3})*(?:\.\d{1,4})?)',
            r'[\$](\d+(?:,\d{3})*(?:\.\d{1,4})?)',
        ]

        for pattern in price_contexts:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                try:
                    price_str = matches[0].replace(',', '')
                    return float(price_str)
                except ValueError:
                    continue

        # Fallback to general price patterns
        for pattern in self.price_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                try:
                    price_str = matches[0].replace(',', '')
                    if 'K' in price_str.upper():
                        return float(price_str.upper().replace('K', '')) * 1000
                    elif 'M' in price_str.upper():
                        return float(price_str.upper().replace('M', '')) * 1000000
                    return float(price_str)
                except ValueError:
                    continue

        return None

    def _extract_target(self, message: str) -> Optional[float]:
        """Extract target price with better accuracy"""
        for pattern in self.target_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                try:
                    target_str = matches[0].replace(',', '')
                    if 'K' in target_str.upper():
                        return float(target_str.upper().replace('K', '')) * 1000
                    return float(target_str)
                except ValueError:
                    continue
        return None

    def _extract_stop_loss(self, message: str) -> Optional[float]:
        """Extract stop loss with better patterns"""
        for pattern in self.stop_loss_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                try:
                    sl_str = matches[0].replace(',', '')
                    if 'K' in sl_str.upper():
                        return float(sl_str.upper().replace('K', '')) * 1000
                    return float(sl_str)
                except ValueError:
                    continue
        return None

    def _extract_confidence(self, message: str) -> Optional[str]:
        """Extract confidence with more patterns"""
        patterns = [
            r'(HIGH|MEDIUM|LOW)\s*(?:CONFIDENCE|PROB)',
            r'(\d{1,3})%',
            r'(⭐{1,5})',
            r'(STRONG|WEAK|MODERATE)'
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)

        return None


class EnhancedTelegramBot:
    """Enhanced Telegram bot with signal sending capabilities"""

    def __init__(self):
        # Load configuration
        self.api_id = int(os.getenv('TELEGRAM_API_ID'))
        self.api_hash = os.getenv('TELEGRAM_API_HASH')
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')  # For sending messages
        self.signal_channel = os.getenv('SIGNAL_CHANNEL', '@your_signal_channel')

        # Initialize components
        self.client = None
        self.bot_client = None
        self.binance = BinanceIntegration(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET')
        )
        self.signal_extractor = EnhancedSignalExtractor()
        self.signals_history = []

        # Monitoring settings
        self.channels_to_monitor = self._load_channels()
        self.min_confidence_threshold = float(os.getenv('MIN_CONFIDENCE', '70'))
        self.enable_auto_trading = os.getenv('ENABLE_AUTO_TRADING', 'false').lower() == 'true'

        logger.info(f"Initialized bot to monitor {len(self.channels_to_monitor)} channels")

    def _load_channels(self) -> List[str]:
        """Load channels from environment"""
        channels_env = os.getenv('CHANNELS', '')
        return [ch.strip() for ch in channels_env.split(',') if ch.strip()]

    async def initialize(self):
        """Initialize Telegram clients"""
        logger.info("🚀 Initializing Telegram clients...")

        # Initialize monitoring client
        session_string = os.getenv('TELEGRAM_SESSION_STRING')
        if session_string:
            self.client = TelegramClient.from_string(
                session=session_string,
                api_id=self.api_id,
                api_hash=self.api_hash
            )
        else:
            self.client = TelegramClient('bitcoin_bot', self.api_id, self.api_hash)

        await self.client.start()

        # Initialize bot client for sending messages
        if self.bot_token:
            from telethon import TelegramClient
            self.bot_client = TelegramClient('bot', self.api_id, self.api_hash)
            await self.bot_client.start(bot_token=self.bot_token)
            logger.info("✅ Bot client initialized for sending signals")

        me = await self.client.get_me()
        logger.info(f"✅ Connected as: {me.first_name} ({me.phone})")

    async def start_monitoring(self):
        """Start comprehensive monitoring"""
        global shutdown_flag

        if not self.client:
            await self.initialize()

        logger.info("🤖 Starting enhanced Bitcoin signal monitoring...")
        logger.info(f"📡 Monitoring: {', '.join(self.channels_to_monitor)}")
        logger.info(f"📢 Sending signals to: {self.signal_channel}")

        # Register message handler
        @self.client.on(events.NewMessage)
        async def enhanced_message_handler(event):
            if shutdown_flag:
                return
            await self._process_enhanced_message(event)

        # Start monitoring loop
        while not shutdown_flag:
            try:
                await self.client.run_until_disconnected()
            except Exception as e:
                logger.error(f"Connection error: {e}")
                await asyncio.sleep(30)

    async def _process_enhanced_message(self, event):
        """Process messages with Binance integration"""
        try:
            channel = await event.get_chat()
            channel_username = getattr(channel, 'username', None)

            if not channel_username or channel_username not in self.channels_to_monitor:
                return

            message_text = event.message.text or ""
            if not message_text:
                return

            # Extract signal
            signal = self.signal_extractor.extract_signal(message_text, channel_username)
            if not signal:
                return

            # Enhance with Binance data
            await self._enhance_with_binance_data(signal)

            # Validate signal quality
            if not self._validate_signal_quality(signal):
                logger.info(f"⚠️ Signal quality too low: {signal.confidence}")
                return

            # Store signal
            self.signals_history.append(signal)

            # Process the signal
            await self._process_enhanced_signal(signal)

        except Exception as e:
            logger.error(f"Error processing enhanced message: {e}")

    async def _enhance_with_binance_data(self, signal: TradingSignal):
        """Enhance signal with real-time Binance data"""
        try:
            # Get current market data
            market_data = await self.binance.get_current_price(signal.symbol.replace('USDT', ''))

            if market_data:
                signal.binance_price = market_data['price']
                signal.volume_24h = market_data['volume']

                # Calculate price difference if signal has price
                if signal.price:
                    signal.price_difference = ((market_data['price'] - signal.price) / signal.price) * 100

                logger.info(f"📊 Enhanced signal with Binance data: Price ${market_data['price']:,.2f}")

        except Exception as e:
            logger.error(f"Error enhancing signal with Binance data: {e}")

    def _validate_signal_quality(self, signal: TradingSignal) -> bool:
        """Validate signal quality before processing"""
        # Check confidence threshold
        if signal.confidence and signal.confidence.replace('%', '').isdigit():
            confidence_num = float(signal.confidence.replace('%', ''))
            if confidence_num < self.min_confidence_threshold:
                return False

        # Validate price reasonableness for Bitcoin
        if signal.binance_price and signal.price:
            price_diff = abs((signal.price - signal.binance_price) / signal.binance_price) * 100
            if price_diff > 10:  # More than 10% difference seems unrealistic
                logger.warning(f"⚠️ Signal price differs from market by {price_diff:.1f}%")
                return False

        return True

    async def _process_enhanced_signal(self, signal: TradingSignal):
        """Process signal with enhanced features"""
        logger.info(f"🚨 ENHANCED SIGNAL: {signal.signal_type} {signal.symbol}")

        # Create formatted message
        formatted_message = self._format_signal_message(signal)

        # Send to Telegram channel
        await self._send_signal_to_telegram(formatted_message)

        # Execute auto-trading if enabled
        if self.enable_auto_trading:
            await self._execute_auto_trade(signal)

        # Log detailed signal
        self._log_detailed_signal(signal)

    def _format_signal_message(self, signal: TradingSignal) -> str:
        """Create beautifully formatted signal message"""
        emoji = "🚀" if signal.signal_type == "BUY" else "📉"

        message = f"{emoji} **{signal.signal_type} SIGNAL** {emoji}\n\n"
        message += f"💎 **{signal.symbol}**\n"

        if signal.binance_price:
            message += f"💰 **Current Price:** ${signal.binance_price:,.2f}\n"

        if signal.price:
            message += f"🎯 **Entry Price:** ${signal.price:,.2f}\n"

        if signal.target_price:
            message += f"🏆 **Target:** ${signal.target_price:,.2f}\n"

        if signal.stop_loss:
            message += f"🛑 **Stop Loss:** ${signal.stop_loss:,.2f}\n"

        if signal.confidence:
            message += f"⭐ **Confidence:** {signal.confidence}\n"

        if signal.price_difference:
            direction = "above" if signal.price_difference > 0 else "below"
            message += f"📊 **Signal price is {abs(signal.price_difference):.1f}% {direction} market**\n"

        if signal.volume_24h:
            message += f"📈 **24h Volume:** {signal.volume_24h:,.0f}\n"

        message += f"\n📡 **Source:** @{signal.channel_name}\n"
        message += f"⏰ **Time:** {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        message += f"\n⚠️ *Always do your own research and manage risk appropriately*"

        return message

    async def _send_signal_to_telegram(self, message: str):
        """Send formatted signal to Telegram channel"""
        try:
            if self.bot_client and self.signal_channel:
                await self.bot_client.send_message(self.signal_channel, message, parse_mode='markdown')
                logger.info(f"📤 Signal sent to {self.signal_channel}")
            else:
                logger.warning("⚠️ Bot client or signal channel not configured")

        except Exception as e:
            logger.error(f"Error sending signal to Telegram: {e}")

    async def _execute_auto_trade(self, signal: TradingSignal):
        """Execute automatic trading (paper trading mode)"""
        try:
            if not self.binance.api_key:
                logger.info("📝 Auto-trading disabled (no API keys)")
                return

            # Calculate position size (2% risk)
            balance = 1000  # Demo balance
            position_size = self.binance.calculate_position_size(
                balance_usdt=balance,
                risk_percentage=2.0,
                entry_price=signal.price or signal.binance_price,
                stop_loss=signal.stop_loss
            )

            if position_size > 0:
                # Place test order
                order_result = self.binance.place_test_order(
                    symbol=signal.symbol,
                    side=signal.signal_type,
                    quantity=position_size,
                    price=signal.price
                )

                if order_result.get('success'):
                    logger.info(f"✅ Auto-trade executed: {order_result}")
                else:
                    logger.error(f"❌ Auto-trade failed: {order_result.get('error')}")

        except Exception as e:
            logger.error(f"Error in auto-trading: {e}")

    def _log_detailed_signal(self, signal: TradingSignal):
        """Log detailed signal information"""
        signal_data = signal.to_dict()
        logger.info(f"📋 Signal Details: {json.dumps(signal_data, indent=2, default=str)}")

        print(f"\n{'=' * 60}")
        print(f"🚨 {signal.signal_type} SIGNAL DETECTED")
        print(f"{'=' * 60}")
        print(f"Symbol: {signal.symbol}")
        print(f"Entry: ${signal.price:,.2f}" if signal.price else "Entry: Market Price")
        print(f"Current: ${signal.binance_price:,.2f}" if signal.binance_price else "Current: N/A")
        print(f"Target: ${signal.target_price:,.2f}" if signal.target_price else "Target: N/A")
        print(f"Stop Loss: ${signal.stop_loss:,.2f}" if signal.stop_loss else "Stop Loss: N/A")
        print(f"Confidence: {signal.confidence}" if signal.confidence else "Confidence: N/A")
        print(f"Source: @{signal.channel_name}")
        print(f"Time: {signal.timestamp}")
        print(f"Raw Message: {signal.raw_message[:150]}...")
        print(f"{'=' * 60}\n")


def signal_handler(signum, frame):
    """Graceful shutdown handler"""
    global shutdown_flag
    logger.info("🛑 Shutdown signal received...")
    shutdown_flag = True


async def main():
    """Enhanced main function"""
    global shutdown_flag

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("🚀 Starting Enhanced Bitcoin Telegram Signal Bot")
    logger.info("=" * 80)

    # Validate environment
    required_vars = ['TELEGRAM_API_ID', 'TELEGRAM_API_HASH', 'CHANNELS']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("\n📋 Required Environment Variables:")
        print("- TELEGRAM_API_ID")
        print("- TELEGRAM_API_HASH")
        print("- CHANNELS (comma-separated)")
        print("- TELEGRAM_SESSION_STRING (after first auth)")
        print("- TELEGRAM_BOT_TOKEN (for sending signals)")
        print("- SIGNAL_CHANNEL (where to send signals)")
        print("- BINANCE_API_KEY (optional, for trading)")
        print("- BINANCE_API_SECRET (optional, for trading)")
        sys.exit(1)

    try:
        # Create and start the enhanced bot
        bot = EnhancedTelegramBot()
        await bot.start_monitoring()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("🛑 Bot shutting down...")


if __name__ == "__main__":
    asyncio.run(main())