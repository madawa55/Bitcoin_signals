import asyncio
import re
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json
import os
from dotenv import load_dotenv

# Required packages: pip install telethon python-dotenv
from telethon import TelegramClient, events
from telethon.tl.types import Channel
from telethon.errors import SessionPasswordNeededError

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bitcoin_signals.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
            r'📈',
            r'🟢',
            r'⬆️',
            r'🚀',
            r'💚'
        ]
        
        self.sell_patterns = [
            r'\b(SELL|SHORT|BEAR|BEARISH|ENTER\s*SHORT)\b',
            r'📉',
            r'🔴',
            r'⬇️',
            r'💥',
            r'❤️'
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

class TelegramSignalMonitor:
    """Main class to monitor Telegram channels for Bitcoin signals"""
    
    def __init__(self):
        # Load configuration from environment variables
        self.api_id = os.getenv('TELEGRAM_API_ID')
        self.api_hash = os.getenv('TELEGRAM_API_HASH')
        self.phone_number = os.getenv('TELEGRAM_PHONE')
        
        # Validate required credentials
        if not self.api_id or not self.api_hash:
            raise ValueError("TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env file")
        
        self.api_id = int(self.api_id)
        self.client = None
        self.signal_extractor = BitcoinSignalExtractor()
        self.signals = []  # Store signals in memory
        
        # Load channels from environment or set defaults
        channels_env = os.getenv('CHANNELS', '')
        if channels_env:
            self.channels_to_monitor = [channel.strip() for channel in channels_env.split(',')]
        else:
            # Default channels - you should replace these with actual channels you have access to
            self.channels_to_monitor = []
            logger.warning("No CHANNELS specified in .env file. Add channels manually using add_channel() method.")
    
    async def initialize(self):
        """Initialize Telegram client and handle authentication"""
        logger.info("Initializing Telegram client...")
        
        self.client = TelegramClient('bitcoin_signals_session', self.api_id, self.api_hash)
        
        try:
            await self.client.start()
            
            # Check if we need to authenticate
            if not await self.client.is_user_authorized():
                logger.info("Authentication required...")
                
                if not self.phone_number:
                    self.phone_number = input("Enter your phone number (with country code, e.g., +1234567890): ")
                
                # Send code
                await self.client.send_code_request(self.phone_number)
                logger.info(f"Verification code sent to {self.phone_number}")
                
                # Get code from user
                code = input('Enter the verification code: ')
                
                try:
                    await self.client.sign_in(self.phone_number, code)
                except SessionPasswordNeededError:
                    # Two-factor authentication
                    password = input('Two-factor authentication enabled. Enter your password: ')
                    await self.client.sign_in(password=password)
            
            # Get user info
            me = await self.client.get_me()
            logger.info(f"Successfully connected as: {me.first_name} {me.last_name or ''} ({me.phone})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram client: {e}")
            raise
    
    async def add_channel(self, channel_username: str) -> bool:
        """Add a channel to monitor"""
        try:
            # Remove @ if present
            channel_username = channel_username.lstrip('@')
            
            entity = await self.client.get_entity(channel_username)
            if isinstance(entity, Channel):
                if channel_username not in self.channels_to_monitor:
                    self.channels_to_monitor.append(channel_username)
                    logger.info(f"Added channel: {entity.title} (@{channel_username})")
                else:
                    logger.info(f"Channel @{channel_username} already being monitored")
                return True
            else:
                logger.error(f"@{channel_username} is not a channel")
                return False
        except Exception as e:
            logger.error(f"Error adding channel @{channel_username}: {e}")
            return False
    
    async def list_available_channels(self):
        """List channels you have access to"""
        logger.info("Listing available channels...")
        
        channels = []
        async for dialog in self.client.iter_dialogs():
            if isinstance(dialog.entity, Channel):
                username = getattr(dialog.entity, 'username', None)
                channels.append({
                    'name': dialog.name,
                    'username': username,
                    'id': dialog.entity.id
                })
        
        if channels:
            logger.info(f"Found {len(channels)} channels:")
            for i, channel in enumerate(channels, 1):
                username_str = f"@{channel['username']}" if channel['username'] else "No username"
                logger.info(f"  {i}. {channel['name']} ({username_str})")
        else:
            logger.info("No channels found")
        
        return channels
    
    async def start_monitoring(self):
        """Start monitoring channels for signals"""
        if not self.client:
            await self.initialize()
        
        if not self.channels_to_monitor:
            logger.error("No channels to monitor. Use add_channel() to add channels or set CHANNELS in .env file.")
            return
        
        logger.info(f"Starting to monitor {len(self.channels_to_monitor)} channels: {', '.join(self.channels_to_monitor)}")
        
        # Verify channel access
        valid_channels = []
        for channel in self.channels_to_monitor:
            try:
                entity = await self.client.get_entity(channel)
                valid_channels.append(channel)
                logger.info(f"✅ Access confirmed for: {entity.title} (@{channel})")
            except Exception as e:
                logger.error(f"❌ Cannot access channel @{channel}: {e}")
        
        self.channels_to_monitor = valid_channels
        
        if not self.channels_to_monitor:
            logger.error("No valid channels to monitor")
            return
        
        # Register event handler for new messages
        @self.client.on(events.NewMessage)
        async def message_handler(event):
            await self._process_message(event)
        
        logger.info("🤖 Bot is now monitoring for Bitcoin signals...")
        logger.info("Press Ctrl+C to stop monitoring")
        
        # Keep the client running
        try:
            await self.client.run_until_disconnected()
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        finally:
            await self.client.disconnect()
    
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
                logger.info(f"🚨 New signal detected: {signal.signal_type} BTC from @{channel_username}")
                
                # Save signal to file
                await self._save_signal(signal)
                
                # Trigger custom callback
                await self._on_signal_received(signal)
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _save_signal(self, signal: TradingSignal):
        """Save signal to JSON file"""
        try:
            # Create signals directory if it doesn't exist
            os.makedirs('signals', exist_ok=True)
            
            filename = f"signals/signals_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Load existing signals
            signals_data = []
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    signals_data = json.load(f)
            
            # Add new signal
            signals_data.append(signal.to_dict())
            
            # Save back to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(signals_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Signal saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
    
    async def _on_signal_received(self, signal: TradingSignal):
        """Custom callback when signal is received - implement your trading logic here"""
        print(f"\n{'='*60}")
        print(f"🚨 NEW BITCOIN SIGNAL DETECTED 🚨")
        print(f"{'='*60}")
        print(f"📊 Signal Type: {signal.signal_type}")
        print(f"💰 Symbol: {signal.symbol}")
        print(f"💵 Current Price: ${signal.price:,.2f}" if signal.price else "💵 Current Price: N/A")
        print(f"🎯 Target Price: ${signal.target_price:,.2f}" if signal.target_price else "🎯 Target Price: N/A")
        print(f"🛑 Stop Loss: ${signal.stop_loss:,.2f}" if signal.stop_loss else "🛑 Stop Loss: N/A")
        print(f"📈 Confidence: {signal.confidence}" if signal.confidence else "📈 Confidence: N/A")
        print(f"📺 Channel: @{signal.channel_name}")
        print(f"⏰ Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📝 Raw Message Preview:")
        print(f"   {signal.raw_message[:200]}{'...' if len(signal.raw_message) > 200 else ''}")
        print(f"{'='*60}\n")
        
        # TODO: Implement your trading logic here
        # Examples:
        # - Send to your trading bot API
        # - Update database
        # - Send email/SMS notifications
        # - Calculate position size
        # - Place orders on exchange
    
    def get_recent_signals(self, hours: int = 24) -> List[TradingSignal]:
        """Get signals from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [signal for signal in self.signals if signal.timestamp >= cutoff_time]
    
    async def get_channel_history(self, channel_username: str, limit: int = 100):
        """Get historical messages from a channel to backtest signal extraction"""
        try:
            entity = await self.client.get_entity(channel_username)
            signals = []
            
            logger.info(f"Analyzing last {limit} messages from @{channel_username}...")
            
            async for message in self.client.iter_messages(entity, limit=limit):
                if message.text:
                    signal = self.signal_extractor.extract_signal(message.text, channel_username)
                    if signal:
                        signal.timestamp = message.date
                        signals.append(signal)
            
            logger.info(f"Found {len(signals)} signals in channel history")
            return signals
            
        except Exception as e:
            logger.error(f"Error getting channel history: {e}")
            return []

# Main execution functions
async def setup_channels(monitor: TelegramSignalMonitor):
    """Interactive setup for adding channels"""
    logger.info("Setting up channels...")
    
    # List available channels
    await monitor.list_available_channels()
    
    while True:
        print(f"\nCurrent monitored channels: {', '.join(monitor.channels_to_monitor) if monitor.channels_to_monitor else 'None'}")
        print("\nOptions:")
        print("1. Add a channel")
        print("2. Start monitoring")
        print("3. Test channel history")
        print("4. Quit")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            channel = input("Enter channel username (without @): ").strip()
            if channel:
                success = await monitor.add_channel(channel)
                if success:
                    print(f"✅ Channel @{channel} added successfully")
                else:
                    print(f"❌ Failed to add channel @{channel}")
        
        elif choice == "2":
            if monitor.channels_to_monitor:
                break
            else:
                print("❌ No channels to monitor. Please add at least one channel.")
        
        elif choice == "3":
            if monitor.channels_to_monitor:
                channel = input(f"Test which channel? ({', '.join(monitor.channels_to_monitor)}): ").strip()
                if channel in monitor.channels_to_monitor:
                    signals = await monitor.get_channel_history(channel, 50)
                    if signals:
                        print(f"\n📊 Found {len(signals)} historical signals:")
                        for i, signal in enumerate(signals[-5:], 1):  # Show last 5
                            print(f"  {i}. {signal.signal_type} at ${signal.price} ({signal.timestamp.strftime('%Y-%m-%d %H:%M')})")
                    else:
                        print("No signals found in recent history")
            else:
                print("No channels added yet")
        
        elif choice == "4":
            return False
        
        else:
            print("Invalid choice")
    
    return True

async def main():
    """Main function"""
    print("🤖 Bitcoin Telegram Signal Monitor")
    print("==================================")
    
    try:
        # Create monitor instance
        monitor = TelegramSignalMonitor()
        
        # Initialize connection
        await monitor.initialize()
        
        # Setup channels interactively if none configured
        if not monitor.channels_to_monitor:
            should_continue = await setup_channels(monitor)
            if not should_continue:
                logger.info("Setup cancelled by user")
                return
        
        # Start monitoring
        logger.info("Starting Bitcoin signal monitoring service...")
        await monitor.start_monitoring()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your .env file")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Print startup information
    print("Starting Bitcoin Signal Monitor...")
    print("Make sure your .env file is configured with:")
    print("- TELEGRAM_API_ID")
    print("- TELEGRAM_API_HASH") 
    print("- TELEGRAM_PHONE (optional)")
    print("- CHANNELS (optional, comma-separated)")
    print()
    
    # Run the application
    asyncio.run(main())