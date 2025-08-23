import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import BaseFilter
import logging
from dotenv import load_dotenv
from db import upload_to_minio, minio_client

load_dotenv(dotenv_path='../.env')


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)



BOT_TOKEN = os.getenv("API_KEY")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

class AllMessagesFilter(BaseFilter):
    async def __call__(self, message: types.Message) -> bool:
        return True

@dp.message(AllMessagesFilter())
async def handle_message(message: types.Message) -> None:
    """Handle incoming messages and download photos/videos."""
    chat_id = message.chat.id
    user = message.from_user
    user_name = user.username or f"{user.first_name} {user.last_name or ''}".strip()
    text = message.text or "No text"

    logger.info(f"Message from {user_name} in chat {chat_id}: {text}")

    if message.photo:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        file_name = f"photo_{message.message_id}_{photo.file_unique_id}.jpg"
        file_data = await bot.download_file(file.file_path)
        file_bytes = file_data.read()
        
        success = await upload_to_minio(file_bytes, file_name)
        if success:
            logger.info(f"Successfully uploaded photo: {file_name}")
        else:
            logger.error(f"Failed to upload photo: {file_name}")

    if message.video:
        video = message.video
        file = await bot.get_file(video.file_id)
        file_name = f"video_{message.message_id}_{video.file_unique_id}.mp4"
        
        # Download file as bytes
        file_data = await file.download_as_bytes()
        
        # Upload to MinIO
        success = await upload_to_minio(file_data, file_name)
        if success:
            logger.info(f"Successfully uploaded video: {file_name}")
        else:
            logger.error(f"Failed to upload video: {file_name}")

async def main() -> None:
    """Main function to run the bot."""
    logger.info("Starting bot...")
    await dp.start_polling(bot, allowed_updates=["message"])

if __name__ == "__main__":
    asyncio.run(main())