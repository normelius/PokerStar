

import numpy as np
import pandas as pd
from math import comb
import contextlib
from operator import attrgetter, itemgetter
import time
from collections import Counter
from itertools import groupby, combinations
from numba import jit, njit
import pyarrow as pa
import pyarrow.parquet as pq


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


@contextlib.contextmanager
def timeit(title = ""):
    tstart = time.time()
    yield
    elapsed = time.time() - tstart
    print(title, elapsed)


class Card():
    def __init__(self, card_string):
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

        return False
    
    def __gt__(self, obj):
        if self.valor > obj.valor:
            return True

        return False

    def __lt__(self, obj):
        if  self.valor < obj.valor:
            return True

        return false

    def card_string(self) -> str:
        return str(self.valor) + self.suit

    def same_suit(self, card) -> bool:
        if self.suit == card.suit:
            return True

        return False

    def valor_difference(self, card) -> int:
        """
        Returns the gap between two cards, e.g., 11D and 5H have gap 5. 
        """
        diff = abs(self.valor - card.valor)
        if diff == 0:
            return diff

        return diff - 1


class Algo():
    def __init__(self):
        colors = np.array(["H", "C", "S", "D"])
        self.all_cards = np.array(sorted(([str(value) + color for color in colors for value
            in range(2, 15)])))
        self.board = None
        self.ranks = pd.read_csv('Algo/ranks.csv', sep = '\t', encoding = 'utf-8',
            usecols = ["rank", "primes", "flush"])
        self.number_of_ranks = self.ranks["rank"].iloc[-1]
        

    def analyze(self, board):
        self.board = board
        print("Chens:", self._chens_formula())
        self._check_hand()
    
    def _longest_consecutive(self, valors):
        """
        Finds the longest consecutive list of valors up to the five top, e.g.,
        [14, 14, 11, 10, 7, 6] will return [11, 10]

        Parameters
        ----------
        valors : list
            List containing all valors currently on the table, hand included.

        Returns
        -------
        consecutive : list
            Returns a list containing the top consecutive valors.

        Examples
        --------
        >>> Algo()._longest_consecutive([13, 12, 11, 10])
        [13, 12, 11, 10]

        >>> Algo()._longest_consecutive([14, 14, 12, 13, 7, 10, 11])
        [14, 13, 12, 11, 10]
        """

        # Drop duplicates since it messes up grouping below. We are only 
        # interested in consecutive integers, don't need duplicates.
        valors = sorted(list(dict.fromkeys(valors)))
        vals = []
        for k, g in groupby(enumerate(valors), lambda ix : ix[0] - ix[1]):
            vals.append(list(map(itemgetter(1), g)))
        
        consecutive = sorted(max(sorted(vals, reverse = True), key = len), 
                reverse = True)

        # Only interested in top five consecutive, since a hand can't be bigger.
        if len(consecutive) > 5:
            return consecutive[0:5]

        return consecutive


    def _display_best_hand(self, hand_msg, hand = None):
        print("Best hand:", hand_msg, end = " ")
        for card in hand:
            print(card, end = " ")

        print("")


    def _check_hand(self):
        """
        Check the current best possible hand available.
        Ex: ['14H', '14S', '11D', '10C', '7C', '6S']
        """
        if self.board:
            cards = self.board.player.cards + self.board.open_cards

        else:
            cards = [Card("14H"), Card("13H"), Card("12H"), Card("11H"),
                    Card("10H"), Card("10C"), Card("13C")]
        
        # Sort in decreasing order, so strongest first.
        cards.sort(key = lambda x: x.valor, reverse = True)
        flush = False
        
        if hand := self._royal_straight_flush(cards):
            flush = True
            self._display_best_hand("Royal straight flush", hand)

        elif hand := self._straight_flush(cards):
            flush = True
            self._display_best_hand("Straight flush", hand)

        elif hand := self._four_oak(cards):
            self._display_best_hand("Four of a kind", hand)

        elif hand := self._full_house(cards):
            self._display_best_hand("Full house", hand)
        
        elif hand := self._flush(cards):
            flush = True
            self._display_best_hand("Flush", hand)
    
        elif hand := self._straight(cards):
            self._display_best_hand("Straight", hand)
        
        elif hand := self._three_oak(cards):
            self._display_best_hand("Three of a kind", hand)

        elif hand := self._two_pair(cards):
            self._display_best_hand("Two pair", hand)

        elif hand := self._pair(cards):
            self._display_best_hand("One pair", hand)

        else:
            hand = [cards[0]]
            self._display_best_hand("High card", hand)
       
        best_hand = self._create_best_hand(cards, hand)
        rank = self._check_rank(best_hand, flush) 
        rank_strength = self._check_rank_strength(rank)
    
    def _check_rank_strength(self, rank):
        """
        Checks the rank strength by calculating the percentile.
        A percentile of example '80%' means that the rank is better than
        80% of all the other ranks.

        Parameters
        ----------
        rank : int
            Integer representing rank to be evaluated.

        Returns
        -------
        pct : float
            Percentile placement of rank of all the (1-7462) ranks.

        Examples
        --------
        >>> Algo()._check_rank_strength(7462)
        0.0
        >>> Algo()._check_rank_strength(1)
        99.99
        >>> Algo()._check_rank_strength(7462/2)
        50.0
        """ 
        rank = 7462 - rank
        ranks = np.arange(7462, 0, -1)
        return round(np.count_nonzero(ranks <= rank) / len(ranks) * 100, 2)

    def _check_rank(self, best_hand, is_flush):
        """
        Checks what rank the current best hand have. All ranks are different
        except for the flushes, hence we check if the hand is a flush.

        Parameters
        ----------
        best_hand : list
            A list representing the best hand, i.e., best five cards you
            currently have.

        is_flush : bool
            A boolean specify whether the best hand is a flush or not.
            This is necessary since all ranks differ, except for the flushes,
            which means the rank will be better if a hand also is a flush.

        Returns
        -------
        rank : int
            Returns an int specifying the rank of the current best hand.

        Examples
        --------
        >>> Algo()._check_rank([Card('14H'), Card('13H'), Card('12H'), 
        ...     Card('11H'), Card('10H')], True)
        1

        >>> Algo()._check_rank([Card('12H'), Card('8H'), Card('5H'), 
        ...     Card('3H'), Card('2H')], True)
        1337

        >>> Algo()._check_rank([Card('7H'), Card('7D'), Card('2C'), 
        ...     Card('2H'), Card('3C')], False)
        3215

        """
        prime_product = self._prime_product(best_hand)
        try:
            return self.ranks.loc[(self.ranks["primes"] == prime_product) &
                (self.ranks["flush"] == is_flush)]["rank"].values[0]

        except IndexError:
            return None


    def _prime_product(self, best_hand):
        """
        Calculates the prime product of the best hand.

        Parameters
        ----------
        best_hand : list
            A list containing the five best cards.

        Returns
        -------
        prime_product : int
            The prime product of the cards. Each valor is represented by a 
            prime, hence the product will all be different, except for the 
            flushes.

        Examples
        --------
        >>> Algo()._prime_product([Card('2H'), Card('2C'), Card('2D'),
        ...     Card('2S'), Card('3H')])
        48

        """
        prime_product = 1
        for card in best_hand:
            prime_product *= card.prime

        return prime_product


    def _create_best_hand(self, cards, hand):
        """
        Creates the best possible hand based on what you currently have
        and all the cards available.

        Parameters
        ----------
        cards : list
            A list containing all cards available, e.g., ['14H', '13H', '12H',
            11H', '10H', '8H', '2H']

        hand : list
            A list containing the hand you currently have, e.g., pair, two pair.

        Returns
        -------
        hand : list
            A list containing the possible best hand you can have, e.g., if you
            have a pair, the last three cards will be the highest three.

        Examples
        --------
        >>> Algo()._create_best_hand([Card("14H"), Card("10C"), Card("8D"),
        ...     Card("7S"), Card("6H"), Card("2H"), Card("2C")], 
        ...     [Card("2H"), Card("2C")])
        [2H, 2C, 14H, 10C, 8D]

        """
        for card in cards:
            if (card not in hand and len(hand) <= 4):
                hand.append(card)

        return hand


    def _royal_straight_flush(self, cards):
        if hand := self._straight_flush(cards):
            if hand[0].valor == 14:
                return hand


    def _straight_flush(self, cards):
        suits = [card.suit for card in cards]
        valors = [card.valor for card in cards]
        flush_suit, suit_count = zip(*Counter(suits).most_common(1))
        consecutive_valors = self._longest_consecutive(valors)
        if len(consecutive_valors) == 5 and suit_count[0] >= 5:
            return [card for card in cards if (card.suit == flush_suit[0] 
                and card.valor in consecutive_valors)]

    def _flush(self, cards):
        suits = [card.suit for card in cards]
        flush_suit, count = zip(*Counter(suits).most_common(1))
        if count[0] >= 5:
            return [card for card in cards if card.suit == flush_suit[0]][0:5]


    def _straight(self, cards):
        valors = [card.valor for card in cards]
        consecutive_valors = self._longest_consecutive(valors)
        if len(consecutive_valors) >= 5:
            consecutive_suits = [card.suit for card in cards if 
                card.valor in consecutive_valors]
            return [card for card in cards if (card.valor in 
                consecutive_valors and card.suit in consecutive_suits)]

    def _full_house(self, cards):
        common, count = self._most_common(cards)
        if count == (3, 2):
            return [card for card in cards if (card.valor == common[0] or
                    card.valor == common[1])]

    def _four_oak(self, cards):
        common, count = self._most_common(cards)
        if count == (4, 1):
            return [card for card in cards if card.valor == common[0]]

    def _three_oak(self, cards):
        common, count = self._most_common(cards)
        if count == (3, 1):
            return [card for card in cards if card.valor == common[0]]

    def _two_pair(self, cards):
        common, count = self._most_common(cards)
        if count == (2, 2):
            return [card for card in cards if (card.valor == common[0] or
                    card.valor == common[1])]
    
    def _pair(self, cards):
        common, count = self._most_common(cards)
        if count == (2, 1) or count == (2,):
            return [card for card in cards if card.valor == common[0]]
    
    def _most_common(self, cards):
        """
        Get the two most common valor and suit, e.g.,
        [(14, 2), (11, 1)]
        """
        kinds = Counter([card.valor for card in cards])
        return zip(*kinds.most_common(2))


    def _chens_formula(self):
        """
        Implements Chen's formula, a way to rank starting hands. Algorithm:
        1. Score the highest card, A = 10, K = 8, Q = 7, J = 6, rest = value / 2.
        2. If pairs, multiply by two, e.g., KK = 8*8=16.
        3. Add 2 points if cards are suited.
        4. Subtract points if gap. No gap = 0, 1 gap = -1, 2 gap = -2, 3 gap = -4
           else -5 points
        5. Add 1 point if 0 or 1 card gap, both cards lower than Q.
        """
        score = 0
        player_card1 = self.board.player.cards[0]
        player_card2 = self.board.player.cards[1]

        highest_card = max(player_card1, player_card2, key = attrgetter('valor'))
        card_scores = {14: 10, 13: 8, 12: 7, 11: 6}

        if highest_card.valor in card_scores:
            score += card_scores[highest_card.valor]

        else:
            score += int(highest_card.valor) / 2.0
        
        if player_card1 == player_card2:
            if player_card1.valor in card_scores:
                score += card_scores[player_card1.valor]

            else:
                score += player_card1.valor / 2
                if score < 5:
                    score = 5
        
        if player_card1.same_suit(player_card2):
            score += 2
        
        gaps = {0: 0, 1: -1, 2: -2, 3: -4, 4: -5, 5: -5, 6: -5, 7: -5, 8: -5,
                9: -5, 10: -5, 11: -5, 12: -5}
        gap = player_card1.valor_difference(player_card2)
        score += gaps[gap]

        if (player_card1.valor < 12 and player_card2.valor < 12 and gap <= 1):
            score += 1

        return score

    
    def _card_statistic(self):
        """
        Computes card probabilities.
        General info:
        Binomial coefficient.
        nCr = (n*(n-1)*n(n-2)*...*(n-r-1)) / r! = n! / r!(n-r)! = math.comb(n, r)
        """
        # Distinct 5 card poker hand.
        self.five_card_combinations = comb(52, 5)
        self.hand_combinations = {
                "single": 1302540, 
                "one pair": 1098240,
                "two pair": 123552, 
                "three kind": 54912, 
                "straight": 10200,
                "flush": 5108, 
                "full house": 3744, 
                "four kind": 624,
                "straight flush": 40
        }

        self.hand_probabilities = {}
        for key, value in self.hand_combinations.items():
            self.hand_probabilities[key] = round(value / self.five_card_combinations, 6)
        
        comb_one_pair = comb(4, 2) * comb(13, 1)
        comb_two_pairs = comb(13, 2) * comb(4, 2) * comb(4, 2)
        comb_three = comb(4, 3) * comb(13, 1)
        comb_four = comb(4, 4) * comb(13, 1)
        comb_full_house = comb(4, 3) * comb(13, 1) * comb(4, 2) * comb(12, 1)
        print(comb_one_pair)
        print(comb_two_pairs)
        print(comb_three)
        print(comb_four)
        print(comb_full_house)

        print(combinations('123', 2))


def main():
    import doctest
    doctest.testmod()

    algo = Algo()
    with timeit():
        algo._check_hand()





if __name__ == '__main__':
    main()

