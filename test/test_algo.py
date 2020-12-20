

import unittest

from pokerstar import Algo, Card


class TestAlgo(unittest.TestCase):
    algo = Algo()
    
    def test_longest_consecutive(self):
        valors = [5, 6, 10, 11, 12, 2, 9]
        out = [12, 11, 10, 9]
        self.assertEqual(self.algo._longest_consecutive(valors), out)
    
    def test_best_hand(self):
        cards = [Card("3H"), Card("3S"), Card("8D"), Card("9D"),
                Card("13S")]
        out = ([Card("3H"), Card("3S"), Card("13S"), Card("9D"),
                Card("8D")], 'One pair')
        self.assertEqual(self.algo._best_hand(cards), out)

        cards = [Card("14H"), Card("13H"), Card("12H"), Card("11H"),
                Card("10H")]
        out = ([Card("14H"), Card("13H"), Card("12H"), Card("11H"),
                Card("10H")], 'Royal straight flush')
        self.assertEqual(self.algo._best_hand(cards), out)

        cards = [Card("10H"), Card("4D"), Card("10D"), Card("4H"),
                Card("10S")]
        out = ([Card("10H"), Card("10S"), Card("10D"), Card("4H"),
                Card("4D")], 'Full house')
        self.assertEqual(self.algo._best_hand(cards), out)


if __name__ == '__main__':
    unittest.main()

