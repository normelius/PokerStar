
import requests
import contextlib

@contextlib.contextmanager
def timeit(title = ""):
    start = time.time()
    yield
    elapsed = time.time() - tstart
    print(title, elapsed)



            

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


class Message():
    def __init__(self):
        self.token = "1450783511:AAGuXC44fOanP8GzhKLMpzpa_Ty0GWaPITc"
        self.id = '1438845336'

    
    def send(self):
        url = """https://api.telegram.org/bot{}/sendMessage?chat_id={}
                &parse_mode=Markdown&text={}""".format(self.token, self.id, 
                        self.message)
        
        response = requests.get(url)
        return response.json()

    def compose(self, components):
        self.message = (
            "{turn} \n"
            "Chens: {chens} \n"
            "Best hand: {best_hand_str} \n"
            "Best hand: {best_hand} \n"
            "Rank: {rank}/7462 {rank_strength}% \n\n"
            "Opponent hands: {opponent_number_hands}\n"
            "Opponent outs: {opponent_number_outs}\n"
            "Opponent hand percentage: {opponent_hand_pct}% \n"
            ).format(
                turn = components["turn"],
                best_hand_str = components["best_hand_str"],
                best_hand = components["best_hand"],
                chens = components["chens"],
                rank = components["player_rank"],
                rank_strength = components["rank_strength"],
                opponent_number_hands = components["opponent_number_hands"],
                opponent_number_outs = components["opponent_number_outs"],
                opponent_hand_pct = components["opponent_hand_pct"])

        print(self.message)




def main():
    read_incoming()


if __name__ == '__main__':
    main()

