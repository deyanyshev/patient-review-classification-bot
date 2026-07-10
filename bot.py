import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from data import load_vocab, encode, MAX_LEN
from model import PaddingMask, AttentionPooling

# ==================== КОНСТАНТЫ ====================
# ID админа
ADMIN_CHAT_ID = 714256029

# Токен бота
BOT_TOKEN = "8857086025:AAEZOYma_IeUsQRGqS4rlkB2TuMmIxaXzoI"

# Пути к файлам модели
CHECKPOINT_PATH = "checkpoints/best_model.keras"
VOCAB_PATH = "checkpoints/vocab.json"

# Состояния для ConversationHandler
WAITING_FOR_REVIEW = 1

LABEL_NAMES = {0: "позитивный", 1: "нейтральный/проблема", 2: "ЧП/критичный"}

# Эмодзи для оценки
EMOJI_MAP = {
    0: "🟢",
    1: "🟡",
    2: "🔴"
}


def load_model_and_vocab():
    """Загрузка модели и словаря"""
    try:
        vocab = load_vocab(VOCAB_PATH)
        model = tf.keras.models.load_model(CHECKPOINT_PATH)
        return vocab, model
    except Exception as e:
        logging.error(f"Ошибка загрузки модели: {e}")
        raise


# Глобальные переменные для модели
vocab, model = load_model_and_vocab()

# ==================== НАСТРОЙКА ЛОГИРОВАНИЯ ====================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ==================== ФУНКЦИИ ОЦЕНКИ ====================
def classify_text(text: str) -> tuple:
    """
    Классифицирует текст с помощью нейросети
    Возвращает: (класс, вероятности, название класса)
    """
    try:
        ids = encode(text, vocab, MAX_LEN)
        x = tf.constant([ids], dtype=tf.int32)
        probs = model.predict(x, verbose=0)[0]
        pred = int(probs.argmax())

        return pred, probs, LABEL_NAMES[pred]
    except Exception as e:
        logger.error(f"Ошибка классификации: {e}")
        return -1, None, "Ошибка классификации"


def format_admin_message(text: str, pred: int, probs: list) -> str:
    """
    Форматирует сообщение для админа с эмодзи
    """
    emoji = EMOJI_MAP.get(pred, "⚪")
    probs_str = ", ".join([f"{i + 1}: {p * 100:.1f}%" for i, p in enumerate(probs)])

    return f"""
{emoji} *НОВЫЙ ОТЗЫВ*

*Оценка нейросети:* {pred + 1} ({LABEL_NAMES[pred]})
*Вероятности:* {probs_str}

*Текст отзыва:*
{text}
"""


# ==================== ОБРАБОТЧИКИ КОМАНД ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    await update.message.reply_text(
        f"👋 Здравствуйте, {user.first_name}!\n\n"
        "Я бот для сбора отзывов о медицинских организациях.\n"
        "Пожалуйста, напишите ваш отзыв, и я передам его руководству.\n\n"
        "📝 Ваш отзыв может содержать:\n"
        "- Впечатления о качестве обслуживания\n"
        "- Замечания или жалобы\n"
        "- Благодарности\n\n"
        "✏️ Напишите ваш отзыв одним сообщением:"
    )
    return WAITING_FOR_REVIEW


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отмена диалога"""
    await update.message.reply_text(
        "❌ Отмена. Если захотите оставить отзыв, нажмите /start"
    )
    return ConversationHandler.END


async def handle_review(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка полученного отзыва"""
    user = update.effective_user
    review_text = update.message.text

    await update.message.reply_text(
        "✅ Спасибо за ваш отзыв! Он был передан руководству для рассмотрения.\n"
        "Мы ценим ваше мнение и стремимся стать лучше! 🏥"
    )

    # Классификация отзыва
    pred, probs, pred_name = classify_text(review_text)

    if pred == -1:
        # Ошибка классификации
        admin_msg = f"""
⚠️ *ОШИБКА КЛАССИФИКАЦИИ*

*От пользователя:* {user.full_name} (@{user.username or "нет"})
*ID:* {user.id}

*Текст:*
{review_text}
"""
        await context.bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=admin_msg,
            parse_mode='Markdown'
        )
    else:
        # Формирование сообщения для админа
        admin_msg = format_admin_message(review_text, pred, probs)

        try:
            await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=admin_msg,
                parse_mode='Markdown'
            )
            logger.info(f"Отзыв от {user.id} отправлен админу. Класс: {pred}")
        except Exception as e:
            logger.error(f"Не удалось отправить сообщение админу: {e}")
            await update.message.reply_text(
                "⚠️ Техническая ошибка при отправке отзыва. Пожалуйста, попробуйте позже."
            )

    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help"""
    await update.message.reply_text(
        "📖 *Как пользоваться ботом:*\n\n"
        "1. Нажмите /start, чтобы начать\n"
        "2. Напишите ваш отзыв о медицинской организации\n"
        "3. Бот поблагодарит вас и передаст отзыв руководству\n\n"
        "Ваш отзыв будет проанализирован нейросетью и передан администратору.\n\n"
        "Команды:\n"
        "/start - начать диалог\n"
        "/cancel - отменить отправку\n"
        "/help - эта справка",
        parse_mode='Markdown'
    )


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    """Запуск бота"""
    application = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            WAITING_FOR_REVIEW: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_review)
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('help', help_command))

    logger.info("Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()