

from __future__ import annotations

import numpy as np
import pandas as pd
from math import comb
import contextlib
from operator import attrgetter, itemgetter
import time
from collections import Counter
from itertools import groupby, combinations

import telegram


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
    def __init__(self, card_string : str):
        self.valor =  int(''.join(x for x in card_string if x.isdigit()))
        self.suit = card_string[-1]
        self.prime = PRIMES[self.valor]
    
    def __str__(self):
        return self.card_string()
    
    def __repr__(self):
        return self.card_string()

    def __eq__(self, obj : Card):
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
        self.all_cards = np.array(sorted(([str(value) + color for color 
            in colors for value in range(2, 15)])))
        self.board = None
        self.ranks = pd.read_csv('Algo/ranks.csv', sep = '\t', 
            encoding = 'utf-8', usecols = ["rank", "primes", "flush"])
        

    def analyze(self, board):
        """
        Update the algo with the latest board and analyze everything possible.
        """
        self.board = board
        num_open_cards = self.board.number_open_cards()

        if num_open_cards == 0:
            telegram.preflop_msg(self._chens())
        
        if num_open_cards >= 3:
            all_cards = self.board.player.cards + self.board.open_cards
            best_hand, hand_str = self._best_hand(all_cards)
            rank = self._check_rank(best_hand) 
            rank_strength = self._check_rank_strength(rank)
            telegram.flop_msg(best_hand, hand_str, rank, rank_strength)

    
    def _longest_consecutive(self, valors : list) -> list:
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

        >>> Algo()._longest_consecutive([14, 14, 13, 12, 5, 2])
        [14, 13, 12]
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


    def _best_hand(self, cards : list) -> (list, str):
        """
        Check the current best possible hand available by comparing the users
        current hand and all cards at the table, compared to all possible hands.
        It uses two different hashtables, one for non-flush and one for flush.
        The idea is that each valor have a different prime number associated
        to it, meaning the hands product will always be different, except 
        for flushes.
        
        Parameters
        ----------
        cards : list
            A list containing all cards to be used. It contains both the open
            cards at the table and the players cards.

        Returns
        -------
        hand : list
            A list containing the best five cards available right now.
        hand_str : str
            String containing a message corresponding to the best hand.

        Exampes
        -------
        >>> Algo()._best_hand([Card('2H'), Card('3H'), Card('3D'), Card('10D'),
        ...     Card('12C')])
        ([3H, 3D, 12C, 10D, 2H], 'One pair')

        >>> Algo()._best_hand([Card('12H'), Card('12C'), Card('14H'), Card('14C'),
        ...     Card('5S')])
        ([14H, 14C, 12H, 12C, 5S], 'Two pair')

        >>> Algo()._best_hand([Card('12H'), Card('11H'), Card('9H'), Card('8H'),
        ...     Card('10H'), Card('4H'), Card('2S')])
        ([12H, 11H, 10H, 9H, 8H], 'Straight flush')
        """
        
        # Sort in decreasing order, so strongest first.
        cards.sort(key = lambda x: x.valor, reverse = True)
        flush = False
        
        if hand := self._royal_straight_flush(cards):
            flush = True
            hand_str = "Royal straight flush"

        elif hand := self._straight_flush(cards):
            flush = True
            hand_str = "Straight flush"

        elif hand := self._four_oak(cards):
            hand_str = "Four of a kind"

        elif hand := self._full_house(cards):
            hand_str = "Full house"
        
        elif hand := self._flush(cards):
            flush = True
            hand_str = "Flush"
    
        elif hand := self._straight(cards):
            hand_str = "Straight"
        
        elif hand := self._three_oak(cards):
            hand_str = "Three of a kind"

        elif hand := self._two_pair(cards):
            hand_str = "Two pair"

        elif hand := self._pair(cards):
            hand_str = "One pair"

        else:
            hand = [cards[0]]
            hand_str = "High card"
       
        best_hand = self._create_best_hand(cards, hand)
        return best_hand, hand_str

    
    def _check_rank_strength(self, rank : int):
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

    def _check_rank(self, best_hand : list):
        """
        Checks what rank the current best hand have. All ranks are different
        except for the flushes, hence we check if the hand is a flush.

        Parameters
        ----------
        best_hand : list
            A list representing the best hand, i.e., best five cards you
            currently have.

        Returns
        -------
        rank : int
            Returns an int specifying the rank of the current best hand.

        Examples
        --------
        >>> Algo()._check_rank([Card('14H'), Card('13H'), Card('12H'), 
        ...     Card('11H'), Card('10H')])
        1

        >>> Algo()._check_rank([Card('12H'), Card('8H'), Card('5H'), 
        ...     Card('3H'), Card('2H')])
        1337

        >>> Algo()._check_rank([Card('7H'), Card('7D'), Card('2C'), 
        ...     Card('2H'), Card('3C')])
        3215

        """
        prime_product = self._prime_product(best_hand)
        if self._is_flush(best_hand):
            return self.ranks.loc[(self.ranks["primes"] == prime_product) &
                (self.ranks["flush"] == True)]["rank"].values[0]
        
        return self.ranks.loc[(self.ranks["primes"] == prime_product) &
                (self.ranks["flush"] == False)]["rank"].values[0]

    
    def _is_flush(self, best_hand : list):
        """
        Checks whether the best hand is a flush or not. It uses Counter
        to get the most common suit, if 5 or over, we know the best hand
        is a flush.

        Parameters
        ----------
        best_hand : list
            A list containing the best hand the user currently have.

        Returns
        -------
        bool
            True if flush has been found, False otherwise.
        """

        suits = [card.suit for card in best_hand]
        flush_suit, count = zip(*Counter(suits).most_common(1))
        if count[0] >= 5:
            return True

        return False


    def _prime_product(self, best_hand : list):
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


    def _create_best_hand(self, cards : list, hand : list) -> list:
        """
        Creates the best possible hand based on what you currently have
        and all the cards available. Observe that this assumes 'cards' param
        are sorted, since hand already containg e.g., pairs, this method
        adds the remaining best cards.

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
        """
        Checks if a royal straight flush exist.

        Examples
        --------
        >>> Algo()._royal_straight_flush([Card("14H"), Card("13H"), Card("12H"), Card("11H"), 
        ...     Card("10H")])
        [14H, 13H, 12H, 11H, 10H]
        """
        if hand := self._straight_flush(cards):
            if hand[0].valor == 14:
                return hand


    def _straight_flush(self, cards):
        """
        Checks if a straight flush exist.

        Examples
        --------
        >>> Algo()._straight_flush([Card("6H"), Card("5H"), Card("4H"), Card("3H"), 
        ...     Card("2H")])
        [6H, 5H, 4H, 3H, 2H]
        """
        suits = [card.suit for card in cards]
        valors = [card.valor for card in cards]
        flush_suit, suit_count = zip(*Counter(suits).most_common(1))
        consecutive_valors = self._longest_consecutive(valors)
        if len(consecutive_valors) == 5 and suit_count[0] >= 5:
            return [card for card in cards if (card.suit == flush_suit[0] 
                and card.valor in consecutive_valors)]

    def _flush(self, cards):
        """
        Checks if a flush exist.

        Examples
        --------
        >>> Algo()._flush([Card("7H"), Card("5H"), Card("4H"), Card("3H"), 
        ...     Card("2H")])
        [7H, 5H, 4H, 3H, 2H]
        """
        suits = [card.suit for card in cards]
        flush_suit, count = zip(*Counter(suits).most_common(1))
        if count[0] >= 5:
            return [card for card in cards if card.suit == flush_suit[0]][0:5]


    def _straight(self, cards):
        """
        Checks if a straight exist.

        Examples
        --------
        >>> Algo()._straight([Card("7H"), Card("6D"), Card("5S"), Card("4C"), 
        ...     Card("3H")])
        [7H, 6D, 5S, 4C, 3H]
        """
        valors = [card.valor for card in cards]
        consecutive_valors = self._longest_consecutive(valors)
        if len(consecutive_valors) >= 5:
            consecutive_suits = [card.suit for card in cards if 
                card.valor in consecutive_valors]
            return [card for card in cards if (card.valor in 
                consecutive_valors and card.suit in consecutive_suits)]

    def _full_house(self, cards):
        """
        Checks if a full house kind exist.

        Examples
        --------
        >>> Algo()._full_house([Card("14H"), Card("14D"), Card("6D"), Card("6H"), 
        ...     Card("6C")])
        [14H, 14D, 6D, 6H, 6C]
        """
        common, count = self._most_common(cards)
        if count == (3, 2):
            return [card for card in cards if (card.valor == common[0] or
                    card.valor == common[1])]

    def _four_oak(self, cards):
        """
        Checks if four of a kind exist.

        Examples
        --------
        >>> Algo()._four_oak([Card("14H"), Card("6S"), Card("6D"), Card("6H"), 
        ...     Card("6C")])
        [6S, 6D, 6H, 6C]
        """
        common, count = self._most_common(cards)
        if count == (4, 1):
            return [card for card in cards if card.valor == common[0]]

    def _three_oak(self, cards):
        """
        Checks if three of a kind exist.

        Examples
        --------
        >>> Algo()._three_oak([Card("14H"), Card("10C"), Card("6D"), Card("6H"), 
        ...     Card("6C")])
        [6D, 6H, 6C]
        """
        common, count = self._most_common(cards)
        if count == (3, 1):
            return [card for card in cards if card.valor == common[0]]

    def _two_pair(self, cards):
        """
        Checks if two pairs exist.

        Examples
        --------
        >>> Algo()._two_pair([Card("14H"), Card("10C"), Card("10D"), Card("8D"), 
        ...     Card("8H")])
        [10C, 10D, 8D, 8H]
        """
        common, count = self._most_common(cards)
        if count == (2, 2):
            return [card for card in cards if (card.valor == common[0] or
                    card.valor == common[1])]
    
    def _pair(self, cards : list) -> list:
        """
        Checks if a pair exist.

        Examples
        --------
        >>> Algo()._pair([Card("14H"), Card("10C"), Card("8D"), Card("2H"), 
        ...     Card("2C")])
        [2H, 2C]
        """
        common, count = self._most_common(cards)
        if count == (2, 1) or count == (2,):
            return [card for card in cards if card.valor == common[0]]
    
    def _most_common(self, cards):
        """
        Get the two most common valor and suit.

        Examples
        --------
        >>> Algo()._most_common([Card("14H"), Card("14C"), Card("8D"), Card("7H"), 
        ...     Card("2C")])
        ((14, 8), (2, 1))
        """
        kinds = Counter([card.valor for card in cards])
        common_cards, cards_count = zip(*kinds.most_common(2))
        return common_cards, cards_count


    def _chens(self):
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


def main():
    import doctest
    doctest.testmod()

    algo = Algo()

    with timeit():
        cards = [Card("14H"), Card("13H"), Card("12H"), Card("11H"),
            Card("10H"), Card("10C"), Card("13C")]
        best_hand, hand_str = algo._best_hand(cards)
        rank = algo._check_rank(best_hand) 
        rank_strength = algo._check_rank_strength(rank)
        #telegram.flop_msg(best_hand, hand_str, rank, rank_strength)





if __name__ == '__main__':
    main()

