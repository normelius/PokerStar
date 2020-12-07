
import requests
import contextlib

@contextlib.contextmanager
def timeit(title = ""):
    start = time.time()
    yield
    elapsed = time.time() - tstart
    print(title, elapsed)



def send_message(msg):
    token = "1450783511:AAGuXC44fOanP8GzhKLMpzpa_Ty0GWaPITc"
    id_ = '1438845336'
    url = """https://api.telegram.org/bot{}/sendMessage?chat_id={}
            &parse_mode=Markdown&text={}""".format(token, id_, msg)
    
    response = requests.get(url)
    return response.json()
            

def preflop_msg(chens):
    """
    Styles the preflop message.
    """
    msg = """Preflop \nChen's: {}
    """.format(chens)
    send_message(msg)

def flop_msg(best_hand, hand_str, rank, rank_strength):
    """
    Styles the flop message.
    """
    msg = """Flop \n{} \nHand {} \nrank {}/7462 | {}%""".format(hand_str, best_hand,
            rank, rank_strength)
    send_message(msg)


def read_incoming():

    url = "https://api.telegram.org/bot1450783511:" + \
            "AAGuXC44fOanP8GzhKLMpzpa_Ty0GWaPITc/getUpdates"

    response = requests.get(url).json()
    try:
        latest_text = response["result"][-1]["message"]["text"].strip().split()
        if len(latest_text) == 1:
            return latest_text[0]

    except (KeyError, IndexError):
        return None


def main():
    read_incoming()


if __name__ == '__main__':
    main()

