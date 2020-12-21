
from random import randint, sample, randrange

PRIMES = {
    2: 2,
    3: 3,
    4: 5,
    5: 7,
    6: 11,
    7: 13,
    8: 17,
    9: 19,
    10: 23,
    11: 29,
    12: 31,
    13: 37,
    14: 41}


class Card():
    def __init__(self, card_string: str):
        self.valor =  int(''.join(x for x in card_string if x.isdigit()))
        self.suit = card_string[-1]
        self.prime = PRIMES[self.valor]

    def __str__(self):
        return self.card_string()
                    
    def __repr__(self):
        return self.card_string()

    def __eq__(self, obj):
        if self.valor == obj.valor:
            return True

    def __gt__(self, obj):
        if self.valor > obj.valor:
            return True

    def __lt__(self, obj):
        if  self.valor < obj.valor:
            return True
    
    def card_string(self) -> str:
        return str(self.valor) + self.suit
    
    def same_suit(self, card) -> bool:
        if self.suit == card.suit:
            return True
                                    
    def valor_difference(self, card) -> int:
        """
        Returns the gap between two cards, e.g., 11D and 5H have gap 5.
        """
        diff = abs(self.valor - card.valor)
        if diff == 0:
            return diff
    
        return diff - 1



class Deck:
    def __init__(self, number_community = 3):
        """
        Representation of a card deck, used in Texas Holdem.

        Parameters
        -----------
        number_community : int
            An int representing how many community cards one should initialize
            with. 3 represent the flop, 4 the turn and 5 the river.
        """

        if not 3 <= number_community <= 5:
            raise ValueError("Param 'number_community' needs to be between" \
                    " 3 and 5")

        self.number_community = number_community
        
        self.reset()
        self._generate_community_cards()
        self._generate_player_cards()

    def _generate_community_cards(self):
        self.community_cards = [self.deck.pop(randrange(len(self.deck))) 
                for _ in range(self.number_community)]

    def _generate_player_cards(self):
        self.player_cards = [self.deck.pop(randrange(len(self.deck))) 
                for _ in range(2)]

    def turn(self):
        if len(self.community_cards) != 3:
            raise ValueError("There needs to be three community cards before" \
                    " the turn.")

        self.community_cards.append(self.deck.pop(randrange(len(self.deck))))

    def river(self):
        if len(self.community_cards) != 4:
            raise ValueError("There needs to be four community cards before" \
                    " the river.")

        self.community_cards.append(self.deck.pop(randrange(len(self.deck))))

    def reset(self):
        self.deck = [Card(str(valor) + suit) for valor in range(2, 15) 
                for suit in ["H", "D", "S", "C"]]
        self._generate_community_cards()
        self._generate_player_cards()







def generate_cards(deck_cards = 3):
    """
    Generates a random set of cards.

    Parameters
    ----------
    deck_cards : int
        Number of cards in the deck, must be between three and five.

    Returns
    -------
    list
        A list containing random cards.
    """
    if not (3 <= deck_cards <= 5):
        raise ValueError("Param 'deck_cards' must be an integer between 3 and 5")

    valors = sample(range(2, 15), cards)
    col = [Card(str(valors[idx]) + ["H", "S", "D", "C"][randint(0, 3)]) 
            for idx in range(cards)]
    return col
    



