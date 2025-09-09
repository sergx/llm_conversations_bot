
import sqlite3
import datetime

DB_PATH = "bot.db"

from config import *


def db_connect():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        telegram_id INTEGER UNIQUE,
        username TEXT,
        first_name TEXT,
        last_name TEXT,
        date_created DATETIME DEFAULT (datetime('now'))
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date_created DATETIME DEFAULT (datetime('now')),
        conversation_name TEXT,
        is_active INTEGER DEFAULT 0,
        model_name TEXT,
        force_audio INTEGER,
        force_text INTEGER,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dialogues (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        date_created DATETIME DEFAULT (datetime('now')),
        message_type TEXT,
        user_voice_filepath TEXT,
        user_voice_transcribed TEXT,
        user_text TEXT,
        llm_text_answer TEXT,
        llm_voice_filepath TEXT,
        llm_image_filepath TEXT,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id)
    )""")
    conn.commit()
    conn.close()


def get_or_create_user(telegram_user):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE telegram_id = ?", (telegram_user.id,))
    row = cur.fetchone()
    if row:
        user_id = row[0]
    else:
        cur.execute("""
            INSERT INTO users (telegram_id, username, first_name, last_name)
            VALUES (?, ?, ?, ?)
        """, (telegram_user.id, telegram_user.username, telegram_user.first_name or "",
              telegram_user.last_name or ""))
        conn.commit()
        user_id = cur.lastrowid
    conn.close()
    return user_id


def create_conversation(user_id, conversation_name=None, model_name=None, set_active=True, force_audio=False, force_text=False):
    model_name = model_name or DEFAULT_MODEL
    conn = db_connect()
    cur = conn.cursor()
    if set_active:
        cur.execute("UPDATE conversations SET is_active = 0 WHERE user_id = ?", (user_id,))
    cur.execute("""
        INSERT INTO conversations (user_id, conversation_name, is_active, model_name, force_audio, force_text)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, conversation_name or "", 1 if set_active else 0, model_name, 1 if force_audio else 0, 1 if force_text else 0))
    conn.commit()
    conv_id = cur.lastrowid
    conn.close()
    return conv_id


def list_conversations(user_id):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            c.id, 
            c.date_created, 
            c.conversation_name, 
            c.is_active, 
            c.model_name,
            COUNT(d.id) AS dialogues_count,
            c.force_audio,
            c.force_text
        FROM conversations c
        LEFT JOIN dialogues d ON d.conversation_id = c.id
        WHERE c.user_id = ?
        GROUP BY c.id
        ORDER BY c.id DESC
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

# cur.execute("""
# CREATE TABLE IF NOT EXISTS dialogues (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     conversation_id INTEGER,
#     date_created TEXT,
#     message_type TEXT,
#     user_voice_filepath TEXT,
#     user_voice_transcribed TEXT,
#     user_text TEXT,
#     llm_text_answer TEXT,
#     llm_voice_filepath TEXT,
#     llm_image_filepath TEXT,
#     FOREIGN KEY(conversation_id) REFERENCES conversations(id)
# )""")
    
def set_active_conversation(user_id, conv_id):
    conn = db_connect()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("UPDATE conversations SET is_active = 0 WHERE user_id = ?", (user_id,))
    cur.execute("UPDATE conversations SET is_active = 1 WHERE id = ? AND user_id = ?", (conv_id, user_id))
    # changed = cur.rowcount
    conn.commit()
    cur.execute("SELECT * FROM dialogues WHERE conversation_id = ? ORDER BY date_created DESC", (conv_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_active_conversation(user_id):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            c.id, 
            c.model_name, 
            c.conversation_name,
            COUNT(d.id) AS dialogues_count,
            force_audio,
            force_text
        FROM conversations c
        LEFT JOIN dialogues d ON d.conversation_id = c.id
        WHERE c.user_id = ? AND c.is_active = 1
        GROUP BY c.id
        LIMIT 1
    """, (user_id,))
    row = cur.fetchone()
    conn.close()
    return row

def get_dialogues_of_conversation(conversation_id):
    conn = db_connect()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT *
        FROM dialogues d
        WHERE d.conversation_id = ?
        ORDER BY d.date_created ASC
    """, (conversation_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def save_dialogue(conversation_id, message_type, **kwargs):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO dialogues (conversation_id, message_type,
                               user_voice_filepath, user_voice_transcribed, user_text,
                               llm_text_answer, llm_voice_filepath, llm_image_filepath)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        conversation_id, message_type,
        kwargs.get("user_voice_filepath"),
        kwargs.get("user_voice_transcribed"),
        kwargs.get("user_text"),
        kwargs.get("llm_text_answer"),
        kwargs.get("llm_voice_filepath"),
        kwargs.get("llm_image_filepath"),
    ))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id