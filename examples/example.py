
import pokerstar as ps
from pokerstar import Card, Deck

def algo_example():
    algo = ps.Algo()
    
    #player_cards = [Card("13H"), Card("13C")]
    #deck_cards = [Card("3H"), Card("14H"), Card("8H"), Card("11D")]

    deck = ps.Deck(3)
    player_cards = deck.player_cards
    community_cards = deck.community_cards

    all_cards = player_cards + community_cards

    chens = algo._chens(player_cards)
    best_hand, best_hand_str = algo._best_hand(all_cards)
    player_rank = algo._check_rank(best_hand)
    rank_strength = algo._check_rank_strength(player_rank)

    opponent_hands, number_hands = algo._get_opponent_hands(community_cards)
    opponent_outs, number_outs = algo._get_opponent_outs(
            opponent_hands, player_rank)
    opponent_hand_pct = algo._hand_percentage(number_hands,
            number_outs)
    

    

def main():
    algo_example()



if __name__ == '__main__':
    main()

