"""Rock, Paper, Scissors (Improved Version)
An enhanced, refactored implementation of the classic hand game.
"""

import random
import time

# Move mappings and rules
MOVES = {
    'R': 'ROCK',
    'P': 'PAPER',
    'S': 'SCISSORS'
}

# Key beats Value(s)
BEATS_MAP = {
    'ROCK': 'SCISSORS',
    'PAPER': 'ROCK',
    'SCISSORS': 'PAPER'
}


def display_intro() -> None:
    """Prints the game intro banner."""
    print("==========================================")
    print("          ROCK, PAPER, SCISSORS           ")
    print("==========================================")
    print("Rules:")
    print("  • Rock beats Scissors")
    print("  • Paper beats Rock")
    print("  • Scissors beats Paper\n")


def get_player_move() -> str | None:
    """Prompts player for input and returns the full move name or None to quit."""
    while True:
        prompt = "Enter move: (R)ock, (P)aper, (S)cissors, or (Q)uit: "
        user_input = input(prompt).strip().upper()

        if user_input == 'Q':
            return None
        if user_input in MOVES:
            return MOVES[user_input]

        print("Invalid input. Please enter R, P, S, or Q.\n")


def countdown_animation() -> None:
    """Plays a brief suspense countdown before revealing computer choice."""
    time.sleep(0.3)
    for i in range(1, 4):
        print(f"{i}...")
        time.sleep(0.25)


def determine_winner(player_move: str, computer_move: str) -> str:
    """Determines winner based on game rules.
    
    Returns: 'tie', 'player', or 'computer'
    """
    if player_move == computer_move:
        return 'tie'
    if BEATS_MAP[player_move] == computer_move:
        return 'player'
    return 'computer'


def play_game() -> None:
    """Main game loop."""
    wins, losses, ties = 0, 0, 0
    display_intro()

    while True:
        print(f"\nScoreboard -> Wins: {wins} | Losses: {losses} | Ties: {ties}")
        
        player_move = get_player_move()
        if player_move is None:
            print("\nThanks for playing! Final Score:")
            print(f"Wins: {wins} | Losses: {losses} | Ties: {ties}")
            break

        print(f"\n{player_move} versus...")
        countdown_animation()

        computer_move = random.choice(list(MOVES.values()))
        print(f"{computer_move}!\n")
        time.sleep(0.3)

        result = determine_winner(player_move, computer_move)

        if result == 'tie':
            print("It's a tie!")
            ties += 1
        elif result == 'player':
            print("🎉 You win!")
            wins += 1
        else:
            print("💻 Computer wins!")
            losses += 1


if __name__ == "__main__":
    play_game()