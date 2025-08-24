from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import asyncpg
import logging
from aiogram import Bot
import asyncio
import dotenv
import os

dotenv.load_dotenv()

logs_path = '/opt/airflow/logs/whale_logs.log' if os.getenv('RUNNING_IN_DOCKER') else '../logs/'

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Bot and database settings (adjust paths for your project)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if BOT_TOKEN is None:
    BOT_TOKEN = os.getenv("API_KEY")

if os.getenv('RUNNING_IN_DOCKER'):
    DATABASE_URL = os.getenv("DATABASE_URL")
else:
    DATABASE_URL = os.getenv("LOCAL_DATABASE_URL")


async def send_emoji_to_chats():
    """Send whale emoji to all unique chats and log the count."""
    bot = Bot(token=BOT_TOKEN)
    try:
        # Query distinct chat_ids
        async with asyncpg.create_pool(DATABASE_URL) as pool:
            async with pool.acquire() as conn:
                chat_ids = await conn.fetch("SELECT DISTINCT chat_id FROM media")
        
        # Send emoji to each chat
        chat_count = 0
        for chat in chat_ids:
            chat_id = chat['chat_id']
            try:
                await bot.send_message(chat_id=chat_id, text="🐳")
                chat_count += 1
            except Exception as e:
                logger.error(f"Failed to send emoji to chat {chat_id}: {e}")
        
        # Log the result
        logger.info(f"Sent {chat_count} whales to {chat_count} chats")
    
    finally:
        await bot.session.close()

def run_async_task():
    """Wrapper to run async send_emoji_to_chats in Airflow."""
    asyncio.run(send_emoji_to_chats())

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'daily_emoji_broadcast',
    default_args=default_args,
    description='Send whale emoji to all chats daily at specified time',
    schedule='0 0 * * *',
    start_date=datetime(2025, 8, 24),
    catchup=False,
) as dag:
    send_emoji_task = PythonOperator(
        task_id='send_emoji_to_chats',
        python_callable=run_async_task,
    )

if __name__ == "__main__":
    run_async_task()