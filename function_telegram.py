
from telegram.ext import ContextTypes, filters
from telegram.constants import MessageLimit

from config import *

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
    
def allowed_users_filter(update):
    return update.effective_chat and update.effective_chat.id in ALLOWED_CHAT_IDS

class AllowedChatsFilter(filters.BaseFilter):
    def __init__(self, allowed_ids):
        self.allowed_ids = allowed_ids

    def filter(self, message):
        return message.chat_id in self.allowed_ids

allowed_chats = AllowedChatsFilter(ALLOWED_CHAT_IDS)