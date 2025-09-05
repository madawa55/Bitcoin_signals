import asyncio
import os
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

# Load environment variables
load_dotenv()

API_ID = os.getenv('TELEGRAM_API_ID')
API_HASH = os.getenv('TELEGRAM_API_HASH')
PHONE = os.getenv('TELEGRAM_PHONE')

async def test_connection():
    """Test Telegram connection and authentication"""
    
    if not API_ID or not API_HASH:
        print("❌ Error: Please set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env file")
        return False
    
    print(f"🔄 Connecting to Telegram...")
    print(f"API ID: {API_ID}")
    print(f"Phone: {PHONE}")
    
    # Create client
    client = TelegramClient('session_name', int(API_ID), API_HASH)
    
    try:
        # Start client
        await client.start()
        
        # Check if we're authorized
        if not await client.is_user_authorized():
            print("📱 Not authorized. Starting authentication...")
            
            # Send code request
            await client.send_code_request(PHONE)
            print(f"📨 Code sent to {PHONE}")
            
            # Get code from user
            code = input('Enter the code you received: ')
            
            try:
                # Sign in with code
                await client.sign_in(PHONE, code)
            except SessionPasswordNeededError:
                # Two-factor authentication enabled
                password = input('Two-factor authentication enabled. Enter your password: ')
                await client.sign_in(password=password)
        
        # Test connection by getting user info
        me = await client.get_me()
        print(f"✅ Successfully connected!")
        print(f"👤 Logged in as: {me.first_name} {me.last_name or ''}")
        print(f"📞 Phone: {me.phone}")
        print(f"🆔 User ID: {me.id}")
        
        # Test getting dialogs (chats/channels)
        print(f"\n📋 Your recent chats/channels:")
        async for dialog in client.iter_dialogs(limit=10):
            chat_type = "Channel" if hasattr(dialog.entity, 'broadcast') and dialog.entity.broadcast else "Chat"
            print(f"  - {dialog.name} ({chat_type})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Telegram: {e}")
        return False
    
    finally:
        await client.disconnect()

if __name__ == "__main__":
    success = asyncio.run(test_connection())
    if success:
        print(f"\n🎉 Connection test successful! You can now run the main Bitcoin signals service.")
    else:
        print(f"\n💡 Please check your credentials and try again.")