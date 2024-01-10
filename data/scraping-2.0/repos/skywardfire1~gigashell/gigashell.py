#!/usr/bin/env python

# Всякие импорты
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat
import platform, os, argparse, distro

def parse_arguments():
  parser = argparse.ArgumentParser( prog='gigashell',
                                    description='Sber GigaChat в твоей консоли!',
                                    epilog='Возрадуемся же!' )
  parser.add_argument( 'request', metavar = 'ЗАПРОС', nargs = '?', default = '', help = 'Запрос к GigaChat' )
  group = parser.add_mutually_exclusive_group()
  group.add_argument( '-s', '--shell', action = 'store_true', help = 'Сгенерировать только финальную команду. Имеет смысл только совместно с запросом' )
# TODO group.add_argument( '-c', '--chat', nargs = 1, action = 'store', help = 'Название для чата' )
# TODO group.add_argument( '-l', '--list-chats', action = 'store_true', help = 'Список чатов' )
  group.add_argument( '-v', '--version', action = 'store_true', help = 'Вывести информацию о версии, и закончить работу' )
  # Здесь возвращаем сразу результат возврата функции
  return parser.parse_args()

# Собираем данные о системе и составляем разогревочный промт
def make_prompt():
  system_os = platform.system()
  system_architecture = platform.machine()
  system_distributive = distro.name()
  distributive_version = distro.version()
  kernel_version = platform.release()
  shell_name = os.readlink('/proc/%d/exe' % os.getppid())

  warmup_prompt = f'Твоя задача отвечать на вопрос пользователя, который работает в операционной системе {system_os}, версия ядра {kernel_version}. Дистрибутив называется {system_distributive} {distributive_version}. Архитектура системы {system_architecture}. Оболочка {shell_name}.'
  # Если включен флаг -s, то скрипт должен вывести только лишь команду, без объяснений. Пока на данный момент работает плохо
  if arguments.shell:
    warmup_prompt = warmup_prompt + ' ' + 'Не пиши дополнительных объяснений. Напиши только одну команду. Нужна только одна команда. Не выводи тэгов code. Команда должна быть готова к выполнению без дополнительных правок. Если тебе недостаточно данных, предоставь наиболее логичное решение. Все перечисленные требования обязательны к выполнению. Без исключений. Все перечисленные требования обязательны к выполнению, без исключений. Ответь на вопрос кратко, одной командой.'
  else:
    warmup_prompt = warmup_prompt + ' ' + 'Отвечай на вопрос развёрнуто, с объяснениями'

  return warmup_prompt

# В функцию надо передать запрос от пользователя
def do_request( system_message, request_text ):
  # Авторизация в сервисе GigaChat
  chat = GigaChat( verify_ssl_certs=False)
  messages = [ SystemMessage( content = system_message ), HumanMessage( content = request_text ) ]
  res = chat( messages )
  print( res.content )
  return 0

if __name__ == '__main__':
  # Парсим аргументы
  arguments = parse_arguments()

  if arguments.version:
    print( 'GigaShell, версия 0.1, 26 ноября 2023' )
    exit

  if not arguments.request == '': do_request( make_prompt(), arguments.request )
