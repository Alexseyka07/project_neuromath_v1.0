import telebot
from colorama import Fore, Back, Style
import os
import filetool
from nn_features import predict

TOKEN = os.environ.get('TELEGRAM_TOKEN')

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start_message(message):
    
    bot.send_message(message.chat.id, 'Привет, в какой задачи вам помочь разобраться?')

@bot.message_handler(content_types=['text'])
def handle_message(message):
    user_input = message.text
    expr_input = ""
    pred = predict(user_input, expr_input)
    answer = filetool.get_rules()[pred[1]]
    bot.send_photo(message.chat.id, answer)

print(f"{Fore.GREEN}[INFO] Bot started{Style.RESET_ALL}")
bot.polling()

