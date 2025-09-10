
import traceback

from pprint import pformat
from telegram import Update
from telegram.ext import ContextTypes, filters
from telegram.constants import MessageLimit

from config import *

from functions_logging import setup_logger
logger = setup_logger(__name__)

async def safe_send_message(context: ContextTypes.DEFAULT_TYPE, chat_id, text, **kwargs):
    chunks = [text[i:i + MessageLimit.MAX_TEXT_LENGTH] 
              for i in range(0, len(text), MessageLimit.MAX_TEXT_LENGTH)]
    for chunk in chunks:
        await context.bot.send_message(chat_id=chat_id, text=chunk, **kwargs)
    # results = []
    # for chunk in chunks:
    #     if args:
    #         new_args = (chat_id, chunk, *args[1:])
    #     else:
    #         new_args = (chat_id,)
    #     new_kwargs = {**kwargs, "text": chunk}
    #     msg = await context.bot.send_message(*new_args, **new_kwargs)
    #     results.append(msg)
    # return results
    
async def bot_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Log the error and send a telegram message if possible."""
    # Log the error before we do anything else
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    # Get the full traceback
    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = ''.join(tb_list)
    
    # Pretty-format the update and context data
    update_str = pformat(update.to_dict(), indent=1, width=80, compact=False) if isinstance(update, Update) else pformat(str(update))
    chat_data_str = pformat(context.chat_data, indent=1, width=80, compact=False)
    user_data_str = pformat(context.user_data, indent=1, width=80, compact=False)
    
    # Log detailed error information with pretty-printed data
    error_message = (
        f"⚠️ Exception while handling update:\n"
        f"Update:\n{update_str}\n\n"
        f"Chat Data:\n{chat_data_str}\n\n"
        f"User Data:\n{user_data_str}\n\n"
        f"{tb_string}"
    )
    
    logger.error(error_message)
    
    # Optional: Send full error to admin
    for dev_chat_id in DEVELOPER_CHAT_IDS:
        if len(error_message) > 4096:  # Telegram message length limit
            for x in range(0, len(error_message), 4096):
                await context.bot.send_message(
                    chat_id=dev_chat_id,
                    text=error_message[x:x+4096]
                )
        else:
            await context.bot.send_message(
                chat_id=dev_chat_id,
                text=error_message
            )