import asyncio
import re
import logging
import os
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from threading import Thread
import signal
import sys

# Required packages: pip install telethon python-dotenv aiofiles
from telethon import TelegramClient, events
from telethon.tl.types import Channel
from telethon.errors import SessionPasswordNeededError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging for cloud environments
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Only console output for cloud
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_flag = False

@dataclass
class TradingSignal:
    """Data class to store trading signal information"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence: Optional[str] = None
    timestamp: datetime = None
    raw_message: str = ""
    channel_name: str = ""
    
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
            'channel_name': self.channel_name
        }

class BitcoinSignalExtractor:
    """Extract trading signals from telegram messages"""
    
    def __init__(self):
        # Common patterns for Bitcoin signals
        self.btc_patterns = [
            r'BTC[/\s]*USD?',
            r'BITCOIN',
            r'₿',
            r'#BTC',
            r'\$BTC'
        ]
        
        # Signal type patterns
        self.buy_patterns = [
            r'\b(BUY|LONG|BULL|BULLISH|ENTER\s*LONG)\b',
            r'📈', r'🟢', r'⬆️', r'🚀', r'💚'
        ]
        
        self.sell_patterns = [
            r'\b(SELL|SHORT|BEAR|BEARISH|ENTER\s*SHORT)\b',
            r'📉', r'🔴', r'⬇️', r'💥', r'❤️'
        ]
        
        # Price patterns
        self.price_patterns = [
            r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:\.\d+)?)[Kk]',  # e.g., 45.5K
        ]
        
        # Target and stop loss patterns
        self.target_patterns = [
            r'(?:TARGET|TP|TAKE\s*PROFIT).*?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'🎯.*?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        self.stop_loss_patterns = [
            r'(?:STOP\s*LOSS|SL|STOP).*?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'🛑.*?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
    
    def extract_signal(self, message: str, channel_name: str = "") -> Optional[TradingSignal]:
        """Extract trading signal from message text"""
        message_upper = message.upper()
        
        # Check if message contains Bitcoin-related keywords
        if not any(re.search(pattern, message_upper, re.IGNORECASE) for pattern in self.btc_patterns):
            return None
        
        # Determine signal type
        signal_type = None
        if any(re.search(pattern, message_upper, re.IGNORECASE) for pattern in self.buy_patterns):
            signal_type = "BUY"
        elif any(re.search(pattern, message_upper, re.IGNORECASE) for pattern in self.sell_patterns):
            signal_type = "SELL"
        
        if not signal_type:
            return None
        
        # Extract prices
        price = self._extract_price(message)
        target_price = self._extract_target(message)
        stop_loss = self._extract_stop_loss(message)
        
        # Extract confidence level
        confidence = self._extract_confidence(message)
        
        return TradingSignal(
            symbol="BTCUSD",
            signal_type=signal_type,
            price=price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            raw_message=message,
            channel_name=channel_name
        )
    
    def _extract_price(self, message: str) -> Optional[float]:
        """Extract current price from message"""
        for pattern in self.price_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                try:
                    price_str = matches[0].replace(',', '')
                    if price_str.endswith('K') or price_str.endswith('k'):
                        return float(price_str[:-1]) * 1000
                    return float(price_str)
                except ValueError:
                    continue
        return None
    
    def _extract_target(self, message: str) -> Optional[float]:
        """Extract target price from message"""
        for pattern in self.target_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[0].replace(',', ''))
                except ValueError:
                    continue
        return None
    
    def _extract_stop_loss(self, message: str) -> Optional[float]:
        """Extract stop loss price from message"""
        for pattern in self.stop_loss_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[0].replace(',', ''))
                except ValueError:
                    continue
        return None
    
    def _extract_confidence(self, message: str) -> Optional[str]:
        """Extract confidence level from message"""
        confidence_patterns = [
            r'(HIGH|MEDIUM|LOW)\s*CONFIDENCE',
            r'(\d{1,3})%\s*CONFIDENCE',
            r'⭐{1,5}'  # Star ratings
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

class CloudTelegramSignalMonitor:
    """Cloud-optimized Telegram signal monitor"""
    
    def __init__(self):
        # Load configuration from environment variables
        self.api_id = os.getenv('TELEGRAM_API_ID')
        self.api_hash = os.getenv('TELEGRAM_API_HASH')
        self.phone_number = os.getenv('TELEGRAM_PHONE')
        
        # Validate required credentials
        if not self.api_id or not self.api_hash:
            raise ValueError("TELEGRAM_API_ID and TELEGRAM_API_HASH must be set as environment variables")
        
        self.api_id = int(self.api_id)
        self.client = None
        self.signal_extractor = BitcoinSignalExtractor()
        self.signals = []  # Store signals in memory
        self.last_signal_time = datetime.now()
        
        # Load channels from environment
        channels_env = os.getenv('CHANNELS', '')
        if channels_env:
            self.channels_to_monitor = [channel.strip() for channel in channels_env.split(',')]
        else:
            raise ValueError("CHANNELS must be set in environment variables")
        
        # Cloud-specific settings
        self.session_string = os.getenv('TELEGRAM_SESSION_STRING', '')
        self.webhook_url = os.getenv('WEBHOOK_URL', '')  # Optional webhook for signals
        
        logger.info(f"Initialized monitor for {len(self.channels_to_monitor)} channels")
    
    async def initialize(self):
        """Initialize Telegram client with cloud-optimized session handling"""
        logger.info("Initializing Telegram client for cloud deployment...")
        
        # Use session string if available, otherwise create new session
        if self.session_string:
            logger.info("Using session string from environment")
            self.client = TelegramClient.from_string(
                session=self.session_string,
                api_id=self.api_id,
                api_hash=self.api_hash
            )
        else:
            logger.info("Creating new session")
            self.client = TelegramClient(
                session='bitcoin_signals_cloud',
                api_id=self.api_id,
                api_hash=self.api_hash
            )
        
        try:
            await self.client.start()
            
            # Handle authentication if needed
            if not await self.client.is_user_authorized():
                if not self.phone_number:
                    raise ValueError("TELEGRAM_PHONE must be set for first-time authentication")
                
                logger.info("Authentication required - this should only happen once")
                await self.client.send_code_request(self.phone_number)
                
                # For cloud deployment, you might need to handle this differently
                # You could use a webhook or manual intervention
                raise RuntimeError("Manual authentication required. Please run locally first to create session.")
            
            # Get user info
            me = await self.client.get_me()
            logger.info(f"Successfully connected as: {me.first_name} ({me.phone})")
            
            # Save session string for future use (print it once for manual copying)
            if not self.session_string:
                session_string = self.client.session.save()
                logger.info("=" * 60)
                logger.info("IMPORTANT: Save this session string as TELEGRAM_SESSION_STRING environment variable:")
                logger.info(session_string)
                logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram client: {e}")
            raise
    
    async def validate_channels(self):
        """Validate access to all channels"""
        logger.info("Validating channel access...")
        
        valid_channels = []
        for channel in self.channels_to_monitor:
            try:
                entity = await self.client.get_entity(channel)
                if isinstance(entity, Channel):
                    valid_channels.append(channel)
                    logger.info(f"✅ Access confirmed: {entity.title} (@{channel})")
                else:
                    logger.warning(f"❌ @{channel} is not a channel")
            except Exception as e:
                logger.error(f"❌ Cannot access @{channel}: {e}")
        
        if not valid_channels:
            raise ValueError("No valid channels found. Check your CHANNELS environment variable.")
        
        self.channels_to_monitor = valid_channels
        logger.info(f"Will monitor {len(valid_channels)} valid channels")
    
    async def start_monitoring(self):
        """Start monitoring channels for signals"""
        global shutdown_flag
        
        if not self.client:
            await self.initialize()
        
        await self.validate_channels()
        
        logger.info("🤖 Starting Bitcoin signal monitoring...")
        logger.info(f"Monitoring channels: {', '.join(self.channels_to_monitor)}")
        
        # Register event handler
        @self.client.on(events.NewMessage)
        async def message_handler(event):
            if shutdown_flag:
                return
            await self._process_message(event)
        
        # Start health check thread
        if os.getenv('ENABLE_HEALTH_CHECK', 'false').lower() == 'true':
            Thread(target=self._health_check_loop, daemon=True).start()
        
        # Main monitoring loop with reconnection logic
        while not shutdown_flag:
            try:
                logger.info("🚀 Bot is now monitoring for Bitcoin signals...")
                await self.client.run_until_disconnected()
            except Exception as e:
                logger.error(f"Connection lost: {e}")
                if not shutdown_flag:
                    logger.info("Attempting to reconnect in 30 seconds...")
                    await asyncio.sleep(30)
                    try:
                        await self.client.connect()
                    except Exception as reconnect_error:
                        logger.error(f"Reconnection failed: {reconnect_error}")
    
    async def _process_message(self, event):
        """Process incoming message and extract signals"""
        try:
            # Get channel information
            channel = await event.get_chat()
            channel_username = getattr(channel, 'username', None)
            
            # Only process messages from monitored channels
            if not channel_username or channel_username not in self.channels_to_monitor:
                return
            
            message_text = event.message.text or ""
            if not message_text:
                return
            
            # Extract signal from message
            signal = self.signal_extractor.extract_signal(message_text, channel_username)
            
            if signal:
                self.signals.append(signal)
                self.last_signal_time = datetime.now()
                
                logger.info(f"🚨 NEW SIGNAL: {signal.signal_type} BTC from @{channel_username}")
                
                # Process the signal
                await self._on_signal_received(signal)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _on_signal_received(self, signal: TradingSignal):
        """Handle new signal - optimized for cloud deployment"""
        
        # Log the signal details
        signal_info = {
            'type': signal.signal_type,
            'symbol': signal.symbol,
            'price': signal.price,
            'target': signal.target_price,
            'stop_loss': signal.stop_loss,
            'confidence': signal.confidence,
            'channel': signal.channel_name,
            'timestamp': signal.timestamp.isoformat()
        }
        
        logger.info(f"Signal Details: {json.dumps(signal_info, indent=2)}")
        
        # Send to webhook if configured
        if self.webhook_url:
            await self._send_webhook(signal)
        
        # Store signal (in cloud environment, consider using database)
        await self._store_signal(signal)
        
        # Print formatted output
        print(f"\n🚨 BITCOIN SIGNAL ALERT 🚨")
        print(f"Signal: {signal.signal_type} {signal.symbol}")
        print(f"Price: ${signal.price:,.2f}" if signal.price else "Price: N/A")
        print(f"Target: ${signal.target_price:,.2f}" if signal.target_price else "Target: N/A")
        print(f"Stop Loss: ${signal.stop_loss:,.2f}" if signal.stop_loss else "Stop Loss: N/A")
        print(f"Channel: @{signal.channel_name}")
        print(f"Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Raw: {signal.raw_message[:150]}...")
        print("=" * 50)
    
    async def _send_webhook(self, signal: TradingSignal):
        """Send signal to webhook URL"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                payload = signal.to_dict()
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Signal sent to webhook successfully")
                    else:
                        logger.warning(f"Webhook responded with status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
    
    async def _store_signal(self, signal: TradingSignal):
        """Store signal (in cloud, consider using a database)"""
        try:
            # For now, just keep in memory and log
            # In production, use a database like PostgreSQL, MongoDB, etc.
            signal_data = signal.to_dict()
            
            # You could send to a database here
            # await database.store_signal(signal_data)
            
            logger.info("Signal stored successfully")
            
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")
    
    def _health_check_loop(self):
        """Health check loop to prevent sleeping on some platforms"""
        while not shutdown_flag:
            try:
                logger.info(f"Health check - Active channels: {len(self.channels_to_monitor)}, "
                           f"Total signals: {len(self.signals)}, "
                           f"Last signal: {(datetime.now() - self.last_signal_time).seconds}s ago")
                time.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(300)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            'total_signals': len(self.signals),
            'channels_monitored': len(self.channels_to_monitor),
            'last_signal_time': self.last_signal_time.isoformat(),
            'uptime_seconds': (datetime.now() - self.last_signal_time).total_seconds()
        }

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    logger.info("Received shutdown signal, stopping gracefully...")
    shutdown_flag = True

async def main():
    """Main function optimized for cloud deployment"""
    global shutdown_flag
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("🤖 Starting Bitcoin Telegram Signal Monitor (Cloud Version)")
    logger.info("=" * 60)
    
    # Validate environment
    required_vars = ['TELEGRAM_API_ID', 'TELEGRAM_API_HASH', 'CHANNELS']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    try:
        # Create and initialize monitor
        monitor = CloudTelegramSignalMonitor()
        await monitor.initialize()
        
        # Start monitoring
        await monitor.start_monitoring()
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        logger.info("Application shutting down...")

if __name__ == "__main__":
    # Environment variable validation
    if not all([os.getenv('TELEGRAM_API_ID'), os.getenv('TELEGRAM_API_HASH')]):
        print("❌ Error: Missing required environment variables")
        print("Set these environment variables in your hosting platform:")
        print("- TELEGRAM_API_ID")
        print("- TELEGRAM_API_HASH")
        print("- TELEGRAM_PHONE (for first setup)")
        print("- CHANNELS (comma-separated channel usernames)")
        print("- TELEGRAM_SESSION_STRING (after first setup)")
        sys.exit(1)
    
    # Run the application
    asyncio.run(main())