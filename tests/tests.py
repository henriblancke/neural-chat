import pandas as pd

from tests.chatbot import ChatBot


def run_test():
    bot = ChatBot()

    fp = open('/assets/tests-messages.csv')
    messages = fp.readlines()
    fp.close()

    results = list()
    for message in messages:
        msg = message.strip()
        resp = bot.respond(msg)

        if isinstance(resp, set):
            resp = "|".join(resp)

        results.append((msg, resp))

    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)


if __name__ == '__main__':
    run_test()
