from __future__ import print_function

import sys
from .chatbot import ChatBot


def test_chatbot():
    bot = ChatBot()

    print("Start chatting...")
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        resp = bot.respond(sentence)
        # resp = resp.replace("_UNK", "").strip().lower()
        sys.stdout.write(">> {}".format(resp))
        print("")
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()


if __name__ == '__main__':
    test_chatbot()
