from __future__ import print_function

import sys

from tests.tests import ChatBot


def decode():
    bot = ChatBot()
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        response = bot.respond(sentence)
        print(response)
        print("> ", end="")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
