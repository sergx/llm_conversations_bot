# sudo systemctl restart llm_conversations_bot.service
# sudo systemctl start llm_conversations_bot.service
# sudo systemctl stop llm_conversations_bot.service
# sudo systemctl status llm_conversations_bot.service

import os
import uuid
import logging
import requests
import subprocess
import shlex
import random
import re
import time
import json
import tiktoken
import grapheme
import asyncio
import glob

# python-telegram-bot
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, ConversationHandler

from db import *
from config import *

from dotenv import load_dotenv
load_dotenv()

from functions_logging import setup_logger
logger = setup_logger(__name__)

from function_telegram import *

from openai_proxy_client import openai_client
client = openai_client()

actions = (
    '/newconv_text',
    '/newconv_audio',
    '/newconv',
    '/convs',
    '/renameconv',
)



# --- OpenAI helpers ---
def count_tokens(model_name, text):
    """Count tokens for a given model."""
    enc = tiktoken.get_encoding(model_name)
    return len(enc.encode(text))

def log_api_cost(model_name, tokens, action=""):
    """Log the estimated cost of an API call."""
    costs = get_model_costs()
    model_cost = costs.get(model_name, {"input": 0, "output": 0})
    input_cost = model_cost["input"] * tokens / 1_000_000
    output_cost = model_cost["output"] * tokens / 1_000_000
    total_cost = input_cost + output_cost
    logger.info(f"{action} ({model_name}): estimated cost ${total_cost:.6f}")

def get_model_costs():
    """Fetch and cache model costs."""
    if os.path.exists(COST_CACHE_FILE):
        mtime = os.path.getmtime(COST_CACHE_FILE)
        if time.time() - mtime < COST_CACHE_TTL:
            with open(COST_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)

    costs = {
        "gpt-4o-mini": {"input": 0.60, "output": 2.40},
        "gpt-4o-mini-tts": {"input": 10.00, "output": 20.00},
        "gpt-4o-mini-chat": {"input": 0.60, "output": 2.40},
    }
    with open(COST_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(costs, f)
    return costs

def log_api_cost(model_name, tokens_or_seconds, action=""):
    """Логирует примерную стоимость запроса по кэшу цен."""
    costs = get_model_costs()
    # Пример структуры: costs["models"]["gpt-4o-mini-transcribe"]["usd_per_unit"]
    model_cost = costs.get("models", {}).get(model_name, {})
    usd_per_unit = model_cost.get("usd_per_unit", 0)
    total_cost = tokens_or_seconds * usd_per_unit
    logger.info(f"{action} ({model_name}): estimated cost ${total_cost:.6f}")

async def chat_completion_get_reply(system_prompt, messages, model_name, context=None, chat_id=None):
    logger.info("chat_completion_get_reply...")
    
    
    await safe_send_message(context, chat_id, f"Запрос обрабатывается [chat_completion]")
        
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    async def run_model(m_name):
        return m_name, await asyncio.to_thread(
            client.chat.completions.create,
            model=m_name,
            messages=messages,
        )

    tasks = []
    # if context is not None and chat_id is not None:
    #     tasks.append(run_model("gpt-5-nano"))
    #     tasks.append(run_model("gpt-5-mini"))

    # главный всегда в конце, чтобы было удобно возвращать
    tasks.append(run_model(model_name))

    results = await asyncio.gather(*tasks)

    # отправляем результаты дополнительных моделей
    if context is not None and chat_id is not None:
        for name, resp in results:
            if name != model_name:  # только дополнительные
                await safe_send_message(context, chat_id, f"{name}:\n\n{resp.choices[0].message.content}")

    # находим главный ответ
    for name, resp in results:
        if name == model_name:
            return resp.choices[0].message.content

async def transcribe_audio(file_path, context=None, chat_id=None):
    logger.info("transcribe_audio...")
    
    if context is not None and chat_id is not None:
        await safe_send_message(context, chat_id, f"Запрос обрабатывается [transcribe_audio]")
        
    model_name = "gpt-4o-mini-transcribe"
    with open(file_path, "rb") as f:
        resp = client.audio.transcriptions.create(model=model_name, file=f)
        # from pydub import AudioSegment
        # audio = AudioSegment.from_file(file_path)
        # seconds = len(audio) / 1000
        # log_api_cost("gpt-4o-mini-transcribe", seconds, "transcribe_audio")
    return resp.text

async def tts_generate(text, out_mp3, model_name="gpt-4o-mini-tts", context=None, chat_id=None):
    logger.info("tts_generate...")

    if context is not None and chat_id is not None:
        await safe_send_message(context, chat_id, f"Запрос обрабатывается [tts_generate]")
    # tts_voices = [
    #     "alloy",
    #     "ash",
    #     "ballad",
    #     "coral",
    #     "echo",
    #     "fable",
    #     "nova",
    #     "onyx",
    #     "sage",
    #     "shimmer",
    # ]
    # voice = random.choice(tts_voices)
    voice = "alloy"
    resp = client.audio.speech.create(
        model=model_name,
        voice=voice,
        input=text,
    )
    with open(out_mp3, "wb") as f:
        f.write(resp.read())
        
    # tokens = len(text)
    # log_api_cost(model_name, tokens, "tts_generate")
    return out_mp3





# --- Telegram handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = get_or_create_user(user)
    chat_id = user.id
    if chat_id not in ALLOWED_CHAT_IDS:
        await update.message.reply_text(
            text = (
                f"Ваш chat_id - <pre>{chat_id}</pre>\n"
                f"Сообщите об этом кому надо."
            ),
            parse_mode='HTML'
        )
        return ConversationHandler.END

    active = get_active_conversation(user_id)
    if not active:
        # create_conversation(user_id, conversation_name="Noname", model_name=DEFAULT_MODEL, set_active=True)
        # conv_id, model_name, conversation_name, dialogues_count = get_active_conversation(user_id)
        # await update.message.reply_text(f"Hello {user.first_name}. Created conversation {conv_id} (model {DEFAULT_MODEL}). Actions:\n{'\n'.join(actions)}" )
        await newconv_command(update, context)
    else:
        conv_id, model_name, conversation_name, dialogues_count, force_audio, force_text = active
        await update.message.reply_text(f"Hello again. Actions:\n{'\n'.join(actions)}\n\nYou have active conversation id {conv_id}: \n[{model_name}] — dialogues_count: {dialogues_count}\n<b>{conversation_name}..</b>", parse_mode="HTML")
    
    return ConversationHandler.END


async def newconv_audio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await newconv_command(update, context, force_audio=True)

async def newconv_text_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await newconv_command(update, context, force_text=True)

async def newconv_command(update: Update, context: ContextTypes.DEFAULT_TYPE, force_audio=False, force_text=False):
    user = update.effective_user
    user_id = get_or_create_user(user)
    conv_id = create_conversation(user_id, conversation_name="Noname", model_name=DEFAULT_MODEL, set_active=True, force_audio=force_audio, force_text=force_text)
    
    text = f"Создано новый диалог id: {conv_id} модель: {DEFAULT_MODEL}"
    
    if force_audio:
        text = f"{text} с ответами в аудио"

    if force_text:
        text = f"{text} с ответами в текстовом виде"
    
    await update.message.reply_text(text)

async def renameconv_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = get_or_create_user(user)
    active = get_active_conversation(user_id)
    if not active:
        return await update.message.reply_text("Не выбран диалог")
    conv_id, model_name, conversation_name, dialogues_count, force_audio, force_text = active
    await update.message.reply_text(f"Укажите название диалога. Текущее название:\n<i>{conversation_name}</i>\n\n/cancel — оставить имя диалога как есть", parse_mode="HTML")
    return 'AWAIT_NEW_NAME'

async def text_to_audio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Пришлите текст на озвучку\n\n/cancel — отменить", parse_mode="HTML")
    return 'AWAIT_TEXT'

async def renameconv_command_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Ок, вы больше не переименовываете диалог.\nВсе что вы напишите дальше может быть использовано против вас (с)", parse_mode="HTML")
    return ConversationHandler.END

async def text_to_audio_command_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Вы больше не в режиме text_to_audio.\nВсе что вы напишите дальше может быть использовано против вас (с)", parse_mode="HTML")
    return ConversationHandler.END

async def conv__renameconv_ON_AWAIT_NEW_NAME(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = get_or_create_user(user)
    active = get_active_conversation(user_id)
    conv_id, model_name, conversation_name, dialogues_count, force_audio, force_text = active
    user_text = update.message.text
    conn = db_connect()
    cur = conn.cursor()
    if grapheme.length(user_text) < 128:
        cur.execute("UPDATE conversations SET conversation_name = ? WHERE id = ?", (user_text,conv_id))
    else:
        await update.message.reply_text(f"Воу, это слишком длинный текст. Прошу сократить до 128 символов.\n\n/cancel — оставить имя диалога как есть")
        return 'AWAIT_NEW_NAME'
    conn.commit()
    conv_id = cur.lastrowid
    conn.close()
    await update.message.reply_text(f"{user_text}")
    return ConversationHandler.END

async def convs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat_id = user.id
    user_id = get_or_create_user(user)
    convs = list_conversations(user_id)
    if not convs:
        await update.message.reply_text("No conversations yet. Use /newconv to create one.")
        return
    lines = []
    for row in convs:
        cid, date_created, conv_name, is_active, model_name, dialogues_count, force_audio, force_text = row
        prefix = f"[is_active] id {cid}" if is_active else f"/switch_{cid}"
        postfix = ""
        if force_audio:
            postfix = " force_audio"
        if force_text:
            postfix = " force_text"
            
        if is_active:
            lines.append(f"<i>{prefix} {model_name} — {dialogues_count}\n<b>«{conv_name or '(no name)'}»</b>{postfix}</i>")
        else:
            lines.append(f"{prefix} {model_name} — {dialogues_count}\n<b>«{conv_name or '(no name)'}»</b>{postfix}")
            
    await safe_send_message(context, chat_id, "\n\n".join(lines), parse_mode="HTML")


async def switch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # print(update.message.text)
    user = update.effective_user
    user_id = get_or_create_user(user)
    chat_id = user.id

    match = re.match(r"^/switch_(\d+)$", update.message.text)
    if not match:
        await update.message.reply_text("Usage: /switch_<conv_id>")
        return

    conv_id = int(match.group(1))
    
    # save_dialogue(conv_id, "voice", user_voice_filepath=local_path, user_voice_transcribed=transcript, llm_text_answer=reply_text, llm_voice_filepath=llm_ogg_path)
    # save_dialogue(conv_id, "text", user_text=user_text, llm_text_answer=reply_text, llm_voice_filepath=None)
    rows = set_active_conversation(user_id, conv_id)
    # print(rows)
    if rows:
        await update.message.reply_text(f"Активирован диалог id {conv_id}\n/convs")
        for row in rows:
            row = dict(row)
            if row['user_voice_transcribed']:
                await safe_send_message(context, chat_id, f"User:\n{row['user_voice_transcribed']}")
                
            if row['user_text']:
                await safe_send_message(context, chat_id, f"User:\n{row['user_text']}")
                
            if row['user_voice_filepath'] and os.path.exists(row['user_voice_filepath']):
                await update.message.reply_audio(audio=open(row['user_voice_filepath'], "rb"))

            if row['llm_text_answer']:
                await safe_send_message(context, chat_id, f"GPT:\n{row['llm_text_answer']}")
                
            if row['llm_voice_filepath']:
                try:
                    filepaths = json.loads(row['llm_voice_filepath'])
                    if not isinstance(filepaths, list):
                        filepaths = [filepaths]
                except json.JSONDecodeError:
                    filepaths = [row['llm_voice_filepath']]

                for path in filepaths:
                    if os.path.exists(path):
                        await update.message.reply_audio(audio=open(path, "rb"))


            time.sleep(1)
                
                
    else:
        await update.message.reply_text(f"Не получилось перейти к диалогу {conv_id} (check id belongs to you).")

def build_message_history(conversation_id):
    dialogues = get_dialogues_of_conversation(conversation_id)
    messages = []

    for row in dialogues:
        if row["user_text"]:
            messages.append({"role": "user", "content": row["user_text"]})
        if row['user_voice_transcribed']:
            messages.append({"role": "user", "content": row["user_voice_transcribed"]})
        if row["llm_text_answer"]:
            messages.append({"role": "assistant", "content": row["llm_text_answer"]})

    return messages

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = get_or_create_user(user)
    chat_id = user.id
    active = get_active_conversation(user_id)
    model_name = DEFAULT_MODEL
    if not active:
        conv_id = create_conversation(user_id, conversation_name="Noname", model_name=model_name, set_active=True)
        conv_id, model_name, conversation_name, dialogues_count, force_audio, force_text = get_active_conversation(user_id)
    else:
        conv_id, model_name, conversation_name, dialogues_count, force_audio, force_text = active

    user_text = update.message.text
    # Save incoming text

    # Build context -- simple: system + user message
    
    messages = build_message_history(conv_id)
    messages.append({"role": "user", "content": user_text})
    system_promt = "You are a wise and highly experienced expert. Your answers should reflect deep knowledge, thoughtful reasoning, and practical wisdom. Communicate clearly, with authority, and in a way that inspires trust. Provide concise, professional, and insightful explanations, avoiding unnecessary simplifications."
    reply_text = await chat_completion_get_reply(system_promt, messages, model_name=model_name, chat_id=chat_id, context=context)

    if dialogues_count == 0:
        conv_name = f"{user_text[0:42]}.."
        # update conversation name
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("UPDATE conversations SET conversation_name = ? WHERE id = ?", (conv_name.strip(), conv_id))
        conn.commit()
        conn.close()
    
    
    if force_audio:
        llm_voice_dir = os.path.join("voices", str(user_id), str(conv_id))
        os.makedirs(llm_voice_dir, exist_ok=True)
        llm_filename = f"{uuid.uuid4().hex}_llm.mp3"
        llm_path = os.path.join(llm_voice_dir, llm_filename)
        await tts_generate(reply_text, llm_path, chat_id=chat_id, context=context)
        
        # send result: text + audio
        await safe_send_message(context, chat_id, f"✅{model_name}:\n{reply_text}\n\n{'\n'.join(actions)}")

        # Convert MP3 → OGG for Telegram voice
        llm_ogg_filename = f"{uuid.uuid4().hex}_llm.ogg"
        llm_ogg_path = os.path.join(llm_voice_dir, llm_ogg_filename)
        
        llm_ogg_paths = await convert_and_split_mp3_to_ogg(llm_path, llm_ogg_path, update)
        
        llm_ogg_paths = json.dumps(llm_ogg_paths)
        
        save_dialogue(conv_id, "voice", user_text=user_text, llm_text_answer=reply_text, llm_voice_filepath=llm_ogg_paths)
        
        # Send as a voice message (round bubble in Telegram)
        # await update.message.reply_audio(audio=open(llm_ogg_path, "rb"), title='title')
    else:
        
        save_dialogue(conv_id, "text", user_text=user_text, llm_text_answer=reply_text)
        await safe_send_message(context, chat_id, f"✅{model_name}:\n{reply_text}\n\n{'\n'.join(actions)}")

async def convert_and_split_mp3_to_ogg(in_mp3, out_dir, update):
    os.makedirs(out_dir, exist_ok=True)
    # convert to ogg and split into 300s chunks
    cmd = f'ffmpeg -y -i "{in_mp3}" -c:a libopus -b:a 64k -ar 48000 -ac 1 -f segment -segment_time 300 "{out_dir}/chunk_%03d.ogg"'
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    ogg_files = []
    # send all chunks one by one
    for ogg_file in sorted(glob.glob(f"{out_dir}/chunk_*.ogg")):
        ogg_files.append(ogg_file)
        await update.message.reply_audio(audio=open(ogg_file, "rb"), title=os.path.basename(ogg_file))
        time.sleep(1)
    return ogg_files

async def conv__text_to_audio_ON_AWAIT_TEXT(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = get_or_create_user(user)
    chat_id = user.id
    user_text = update.message.text
    if not user_text:
        await update.message.reply_text("Текста нет, но вы держитесь.")
        return
    
    llm_voice_dir = os.path.join("voices", str(user_id), "text_to_audio")
    os.makedirs(llm_voice_dir, exist_ok=True)
    llm_filename = f"{uuid.uuid4().hex}_llm.mp3"
    llm_path = os.path.join(llm_voice_dir, llm_filename)
    await tts_generate(user_text, llm_path, chat_id=chat_id, context=context)
    
    llm_ogg_filename = f"{uuid.uuid4().hex}_llm.ogg"
    llm_ogg_path = os.path.join(llm_voice_dir, llm_ogg_filename)
    
    llm_ogg_paths = await convert_and_split_mp3_to_ogg(llm_path, llm_ogg_path, update)
    return await text_to_audio_command_cancel(update, context)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = get_or_create_user(user)
    chat_id = user.id
    voice = update.message.voice or update.message.audio
    if not voice:
        await update.message.reply_text("No voice found in message.")
        return

    active = get_active_conversation(user_id)
    if not active:
        conv_id = create_conversation(user_id, conversation_name="Noname", model_name=DEFAULT_MODEL, set_active=True)
        model_name = DEFAULT_MODEL
    else:
        conv_id, model_name, conversation_name, dialogues_count, force_audio, force_text = active

    file = await context.bot.get_file(voice.file_id)
    user_voice_dir = os.path.join("voices", str(user_id), str(conv_id))
    os.makedirs(user_voice_dir, exist_ok=True)
    local_filename = f"{uuid.uuid4().hex}_user.ogg"
    local_path = os.path.join(user_voice_dir, local_filename)
    await file.download_to_drive(local_path)

    # Transcribe
    try:
        await update.message.reply_text("Transcribing your voice...")
        transcript = await transcribe_audio(local_path, chat_id=chat_id, context=context)
        await safe_send_message(context, chat_id, f"User transcript:\n\n{transcript}")
    except Exception as e:
        logger.exception("Transcription failed: %s", e)
        await update.message.reply_text("Failed to transcribe audio.")
        return


    # Send transcript to LLM
    messages = build_message_history(conv_id)
    messages.append({"role": "user", "content": transcript})
    
    await update.message.reply_text("Ожидание текстового ответа от LLM...")
    try:
        system_promt = "You are a wise and highly experienced expert. Your answers should reflect deep knowledge, thoughtful reasoning, and practical wisdom. Communicate clearly, with authority, and in a way that inspires trust. Provide concise, professional, and insightful explanations, avoiding unnecessary simplifications. Your output will be processed by text-to-speech, so avoid using emojis, code blocks, formulas, or any elements that may not be suitable for speech synthesis. Always respond in plain, natural language."
        reply_text = await chat_completion_get_reply(system_promt, messages, model_name=model_name, chat_id=chat_id, context=context)
    except Exception as e:
        logger.exception("LLM failed: %s", e)
        await update.message.reply_text("Failed to get LLM response.")
        return

    conv_name = f"{transcript[0:42]}.."
    # update conversation name
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE conversations SET conversation_name = ? WHERE id = ?", (conv_name.strip(), conv_id))
    conn.commit()
    conn.close()

    # # Save LLM text answer
    # save_dialogue(conv_id, "voice", user_voice_filepath=local_path, user_voice_transcribed=transcript, llm_text_answer=reply_text)

    # Generate TTS audio
    try:
        await safe_send_message(context, chat_id, f"✅{model_name}\n\n{reply_text}\n\n{'\n'.join(actions)}")
        if force_text:
            save_dialogue(conv_id, "text", user_voice_filepath=local_path, user_voice_transcribed=transcript, llm_text_answer=reply_text)
        else:
            llm_voice_dir = os.path.join("voices", str(user_id), str(conv_id))
            os.makedirs(llm_voice_dir, exist_ok=True)
            llm_filename = f"{uuid.uuid4().hex}_llm.mp3"
            llm_path = os.path.join(llm_voice_dir, llm_filename)
            await tts_generate(reply_text, llm_path, chat_id=chat_id, context=context)

            # Convert MP3 → OGG for Telegram voice
            llm_ogg_filename = f"{uuid.uuid4().hex}_llm.ogg"
            llm_ogg_path = os.path.join(llm_voice_dir, llm_ogg_filename)
            
            llm_ogg_paths = await convert_and_split_mp3_to_ogg(llm_path, llm_ogg_path, update)
            
            llm_ogg_paths = json.dumps(llm_ogg_paths)
            
            
            save_dialogue(conv_id, "voice", user_voice_filepath=local_path, user_voice_transcribed=transcript, llm_text_answer=reply_text, llm_voice_filepath=llm_ogg_paths)
            
            # Send as a voice message (round bubble in Telegram)
            # await update.message.reply_audio(audio=open(llm_ogg_path, "rb"), title='title')
    except Exception as e:
        logger.exception("TTS generation or send failed: %s", e)
        await safe_send_message(context, chat_id, f"✅{model_name}\n\n{reply_text}\n\n{'\n'.join(actions)}")

# --- main ---

conversations = {
    'renameconv' : ConversationHandler(
        entry_points=[
            CommandHandler("renameconv", renameconv_command)
        ],
        states={
            'AWAIT_NEW_NAME': [
                # MessageHandler(filters.Regex("^Назад$"), command_handler__start),
                CommandHandler("cancel", renameconv_command_cancel),
                MessageHandler(filters.TEXT & ~filters.COMMAND, conv__renameconv_ON_AWAIT_NEW_NAME)
                ],
        },
        fallbacks=[
            CommandHandler("start", start_command),
        ],
        # per_message=True,
    ),
    'text_to_audio' : ConversationHandler(
        entry_points=[
            CommandHandler("text_to_audio", text_to_audio_command)
        ],
        states={
            'AWAIT_TEXT': [
                # MessageHandler(filters.Regex("^Назад$"), command_handler__start),
                CommandHandler("cancel", text_to_audio_command_cancel),
                MessageHandler(filters.TEXT & ~filters.COMMAND, conv__text_to_audio_ON_AWAIT_TEXT)
                ],
        },
        fallbacks=[
            CommandHandler("start", start_command),
        ],
        # per_message=True,
    ),
}



def main():
    
    os.makedirs(f"voices", exist_ok=True)
    init_db()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(conversations["renameconv"])
    app.add_handler(conversations["text_to_audio"])
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("newconv", newconv_command, filters=filters.User(ALLOWED_CHAT_IDS)))
    app.add_handler(CommandHandler("newconv_audio", newconv_audio_command, filters=filters.User(ALLOWED_CHAT_IDS)))
    app.add_handler(CommandHandler("newconv_text", newconv_text_command, filters=filters.User(ALLOWED_CHAT_IDS)))
    app.add_handler(CommandHandler("convs", convs_command, filters=filters.User(ALLOWED_CHAT_IDS)))
    app.add_handler(MessageHandler(filters.Regex(r"^/switch_\d+$") & filters.User(ALLOWED_CHAT_IDS), switch_command))

    # text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.User(ALLOWED_CHAT_IDS), handle_text))
    # voice or audio
    app.add_handler(MessageHandler((filters.VOICE | filters.AUDIO) & filters.User(ALLOWED_CHAT_IDS), handle_voice))

    logger.info("Starting bot...")
    app.add_error_handler(bot_error_handler)
    app.run_polling(timeout=300)


if __name__ == "__main__":
    if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
        print("Please set TELEGRAM_TOKEN and OPENAI_API_KEY environment variables.")
        raise SystemExit(1)
    
    #get_model_costs()
    main()
