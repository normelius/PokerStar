# PokerStar 2000

World's best pokerbot.

## Installation

Just clone the repo, cd into it and install.
```bash
git clone https://github.com/normelius/PokerStar.git
cd /PokerStar
pip3 install .
```

Or we could just
```bash
pip3 install git+https://github.com/normelius/PokerStar.git
```

## Usage

So, what can we do at the moment?
```python
# Let's create a random deck containing three community card.
# We use the Deck class for simulation. In reality, we would obtain the cards
# from the pokerstars client.
>>> deck = pokerstar.Deck(3)
>>> player_cards = deck.player_cards
>>> print(player_cards)
[13H, 13C]

>>> community_cards = deck.community_cards
>>> print(community_cards)
[3H, 14H, 8H, 11D]
>>> all_cards = player_cards + community_cards

# Initialize an algo object so we can do some fun.
>>> algo = pokerstar.Algo()

# What is the best current hand we have?
>>> best_hand, best_hand_str = algo.best_hand(all_cards)
>>> print(best_hand)
[13H, 13C, 14H, 11D, 8H], "One Pair"

# What score does our best hand have? This is based on the fact that there exist 7462 
# unique hands in total.
>>> rank = algo.rank(best_hand)
>>> print(rank)
3558

# How strong is this rank compared to the other possible ranks?
>>> rank_strength = algo.rank_strength(rank)
>>> print(rank_strength)
52.32


# What can we do with our opponents cards?
# Since we know that our opponents will always use at least three of the 
# community cards. We can create all possible combinations of three of the community 
# cards, and check each and every combination in our DataFrame. Thus we know
# the possible hands they could have.
>>> opponent_hands, number_opponent_ hands = algo.opponent_hands(community_cards)
>>> print(number_hands) 
292

# How many of these are better than our hand?
>>> opponent_outs, num_opponent_outs = algo.opponent_outs(opponent_hands, rank)
>>> print(number_opponent_outs)
82

# How many % of the opponents possible hands are better than ours?
# If we get a really low number, there is a big chance they are bluffing,
# since we can be confident we have a better hand. However, this doesn't
# take into account the possible hands they might get!
>>> opponent_hand_pct = algo.opponent_hand_percentage(num_opponent_hands,
num_opponent_outs)
>>> print(opponent_hand_pct)
28.1

```
Pokerstar also has a whole suit for obtaining the data in realtime from the pokerstars client. This is however a whole other ballpark in documenting. This also needs rework to be more user-friendly, since now it works for me.

## Todo
- Come up with more stuff for the Algo part.
- Make Board pretty and more user-friendly. 
- Write Display class for sending updates both to the console and telegram.


## License
[MIT](https://choosealicense.com/licenses/mit/)

