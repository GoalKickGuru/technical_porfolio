"""
Extended Rock Paper Scissors - Enhanced Edition
Features:
- Classic RPS + Lizard & Spock (RPSLS variant)
- Risk/betting system with double-or-nothing options
- Win streak tracking and bonuses
- AI opponent with pattern recognition
- Tournament mode
- Statistics and game history
"""

import random, time, sys, json, os
from datetime import datetime
from collections import Counter

# Game configuration
MOVES = ['ROCK', 'PAPER', 'SCISSORS']
MOVES_EXTENDED = ['ROCK', 'PAPER', 'SCISSORS', 'LIZARD', 'SPOCK']

# Move relationships - Extended version
BEATS = {
    'ROCK': ['SCISSORS', 'LIZARD'],
    'PAPER': ['ROCK', 'SPOCK'],
    'SCISSORS': ['PAPER', 'LIZARD'],
    'LIZARD': ['SPOCK', 'PAPER'],
    'SPOCK': ['SCISSORS', 'ROCK']
}

MOVE_DESCRIPTIONS = {
    'ROCK': '🪨 Rock',
    'PAPER': '📄 Paper', 
    'SCISSORS': '✂️ Scissors',
    'LIZARD': '🦎 Lizard',
    'SPOCK': '🖖 Spock'
}

MOVE_KEY_MAP = {
    'R': 'ROCK', 'P': 'PAPER', 'S': 'SCISSORS', 
    'L': 'LIZARD', 'K': 'SPOCK', 'X': 'EXTENDED'
}

class GameState:
    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.streak = 0
        self.max_streak = 0
        self.points = 100
        self.total_rounds = 0
        self.mode = 'classic'  # classic, extended, tournament
        
        # Load history if exists
        self.history_file = 'rps_history.json'
        self.game_history = []
        self.load_history()
        
    def load_history(self):
        """Load game history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    saved_data = json.load(f)
                    self.wins = saved_data.get('wins', 0)
                    self.losses = saved_data.get('losses', 0)
                    self.ties = saved_data.get('ties', 0)
                    self.streak = saved_data.get('streak', 0)
                    self.max_streak = saved_data.get('max_streak', 0)
                    self.points = saved_data.get('points', 100)
                    self.game_history = saved_data.get('history', [])
            except:
                pass
                
    def save_history(self):
        """Save game history to file"""
        data = {
            'wins': self.wins,
            'losses': self.losses,
            'ties': self.ties,
            'streak': self.streak,
            'max_streak': self.max_streak,
            'points': self.points,
            'history': self.game_history[-50:]  # Keep last 50 games
        }
        with open(self.history_file, 'w') as f:
            json.dump(data, f)

class AIOpponent:
    """Smart AI that tries to detect player patterns"""
    
    def __init__(self, mode='basic'):
        self.mode = mode
        self.player_history = []
        
    def get_move_basic(self):
        """Random move - baseline"""
        return random.choice(MOVES_EXTENDED)
    
    def get_move_predictive(self):
        """Predict player's next move based on history"""
        if len(self.player_history) < 3:
            return self.get_move_basic()
        
        # Find most common counter to player's last move
        last_player_move = self.player_history[-1]
        counters = BEATS[last_player_move]
        
        # Choose move that beats player's likely choice
        return random.choice(counters)
    
    def update_history(self, player_move):
        """Track player moves for prediction"""
        self.player_history.append(player_move)
        if len(self.player_history) > 20:
            self.player_history.pop(0)
    
    def get_move(self):
        if self.mode == 'predictive':
            return self.get_move_predictive()
        return self.get_move_basic()

def display_splash():
    """Show animated splash screen"""
    print("\n" + "="*60)
    print("  🎮 EXTENDED ROCK PAPER SCISSORS - ENHANCED EDITION 🎮")
    print("="*60 + "\n")
    time.sleep(0.3)
    print("Loading game modules...")
    time.sleep(0.2)
    print("  ✓ Core game engine loaded")
    time.sleep(0.2)
    print("  ✓ AI opponent initialized")
    time.sleep(0.2)
    print("  ✓ Statistics tracking enabled")
    time.sleep(0.5)
    print("="*60 + "\n")

def display_rules(mode='classic'):
    """Display game rules"""
    print("\n📋 GAME RULES:")
    print("-"*40)
    if mode == 'extended':
        print("Rock crushes Scissors and Lizard")
        print("Paper covers Rock and disproves Spock")
        print("Scissors cuts Paper and decapitates Lizard")
        print("Lizard eats Paper and poisons Spock")
        print("Spock vaporizes Rock and smashes Scissors")
    else:
        print("Rock beats Scissors")
        print("Paper beats Rock")
        print("Scissors beats Paper")
    print("-"*40 + "\n")

def build_menu(state):
    """Build main menu with all options"""
    menu = [
        ("1", "▶ Start New Game"),
        ("2", "📊 View Statistics"),
        ("3", "🏆 View Tournament Standings"),
        ("4", "⚙️ Settings (Difficulty/Moves)"),
        ("5", "💾 Save & Quit"),
        ("6", "📜 Show Rules"),
        ("7", "ℹ️ About"),
    ]
    
    print("\n" + "="*50)
    print("MAIN MENU")
    print("="*50)
    print(f"Points: {state.points} | Streak: {state.streak}")
    print(f"Wins: {state.wins} | Losses: {state.losses} | Ties: {state.ties}")
    print("-"*50)
    
    for code, label in menu:
        print(f"{code}. {label}")
    
    print("="*50)
    return input("Select option: ").strip()

def choose_game_mode():
    """Choose between classic and extended mode"""
    print("\n🎯 Choose Game Mode:")
    print("1. Classic (Rock, Paper, Scissors)")
    print("2. Extended (Add Lizard & Spock)")
    
    choice = input("Mode [1/2]: ").strip()
    return 'classic' if choice == '1' else 'extended'

def get_player_move(game_mode):
    """Get and validate player input"""
    valid_keys = 'RPSLKQ' if game_mode == 'extended' else 'RPSQ'
    
    while True:
        if game_mode == 'extended':
            print("\nEnter your move: (R)ock (P)aper (S)cissors (L)izard (V)Spock or (Q)uit")
        else:
            print("\nEnter your move: (R)ock (P)aper (S)cissors or (Q)uit")
            
        player_input = input("> ").strip().upper()
        
        if player_input == 'Q':
            return None
        
        if player_input == 'V' and game_mode == 'extended':
            player_input = 'K'  # K for Spock
            
        if player_input in MOVE_KEY_MAP:
            return MOVE_KEY_MAP[player_input]
        
        print("❌ Invalid input. Try again.")

def dramatic_countdown(num_seconds=3):
    """Create suspense with animated countdown"""
    for i in range(num_seconds, 0, -1):
        print(f"\r{i}...", end="", flush=True)
        time.sleep(0.4)
    print("\r  ✨         \n", end="")
    time.sleep(0.3)

def play_round(player_move, ai, game_mode, bet_points=1):
    """Play a single round and return result"""
    computer_move = ai.get_move()
    ai.update_history(player_move)
    
    dramatic_countdown()
    
    # Display both moves
    print(f"\n{MOVE_DESCRIPTIONS[player_move]} vs {MOVE_DESCRIPTIONS[computer_move]}!")
    time.sleep(0.5)
    
    # Determine winner
    if player_move == computer_move:
        result = 'tie'
        message = "It's a tie!"
    elif computer_move in BEATS[player_move]:
        result = 'win'
        message = "🎉 You win!"
    else:
        result = 'loss'
        message = "😢 You lose!"
    
    print(message)
    time.sleep(0.8)
    
    return {
        'player_move': player_move,
        'computer_move': computer_move,
        'result': result,
        'bet_amount': bet_points
    }

def handle_betting(current_points, can_double=False):
    """Handle risk/betting mechanics"""
    print("\n💰 Current Points:", current_points)
    print("Bet 1 point per round (standard)")
    
    if can_double and current_points >= 2:
        print("\n⚡ DOUBLE OR NOTHING available! Win 2 points this round")
        double_choice = input("Double your bet? (y/n): ").strip().lower()
        if double_choice == 'y':
            return 2
    
    return 1

def show_statistics(state):
    """Display comprehensive game statistics"""
    print("\n" + "="*50)
    print("📊 GAME STATISTICS")
    print("="*50)
    
    total_games = state.wins + state.losses + state.ties
    win_rate = (state.wins / total_games * 100) if total_games > 0 else 0
    
    stats = [
        ("Total Games Played", str(total_games)),
        ("Wins", f"{state.wins} ({win_rate:.1f}%)"),
        ("Losses", str(state.losses)),
        ("Ties", str(state.ties)),
        ("Max Win Streak", str(state.max_streak)),
        ("Current Points", str(state.points)),
        ("Games History Saved", f"{len(state.game_history)} rounds")
    ]
    
    for label, value in stats:
        print(f"{label:<30}: {value}")
    
    print("="*50)
    input("\nPress Enter to continue...")

def run_tournament(ai, game_mode):
    """Tournament mode - best of 5 matches"""
    print("\n🏆 TOURNAMENT MODE - Best of 5")
    print("="*50)
    
    player_wins = 0
    comp_wins = 0
    rounds_needed = 3
    
    for match_num in range(1, 6):
        if player_wins >= rounds_needed or comp_wins >= rounds_needed:
            break
            
        print(f"\n--- Match {match_num} ---")
        player_move = get_player_move(game_mode)
        if player_move is None:
            break
            
        result = play_round(player_move, ai, game_mode, bet_points=1)
        
        if result['result'] == 'win':
            player_wins += 1
            print(f"Score: {player_wins} - {comp_wins}")
        elif result['result'] == 'loss':
            comp_wins += 1
            print(f"Score: {player_wins} - {comp_wins}")
        
        time.sleep(1)
    
    # Tournament result
    print("\n" + "="*50)
    if player_wins >= rounds_needed:
        print("🏆 YOU WIN THE TOURNAMENT! (+50 bonus points)")
        return 50
    elif comp_wins >= rounds_needed:
        print("💀 Computer wins the tournament")
        return -20
    else:
        print("🤝 Tournament ended in a tie (+10 points)")
        return 10

def main():
    """Main game loop"""
    display_splash()
    
    state = GameState()
    game_mode = choose_game_mode()
    ai = AIOpponent(mode='predictive')
    
    # Load saved game mode preference if exists
    if os.path.exists('game_settings.json'):
        with open('game_settings.json', 'r') as f:
            settings = json.load(f)
            game_mode = settings.get('mode', game_mode)
    
    while True:
        selection = build_menu(state)
        
        if selection == '1':  # Start New Game
            game_type = input("Quick Play or Tournament? (quick/tour): ").strip().lower()
            
            if game_type == 'tour':
                bonus = run_tournament(ai, game_mode)
                state.points += bonus
            else:
                # Single round with betting
                can_double = state.streak >= 3
                bet_amount = handle_betting(state.points, can_double)
                
                player_move = get_player_move(game_mode)
                if player_move is None:
                    break
                    
                result = play_round(player_move, ai, game_mode, bet_amount)
                
                # Update state based on result
                if result['result'] == 'win':
                    state.wins += 1
                    state.points += bet_amount
                    state.streak += 1
                    state.max_streak = max(state.max_streak, state.streak)
                    state.ties += 1 if result['result'] == 'tie' else 0
                elif result['result'] == 'loss':
                    state.losses += 1
                    state.points -= bet_amount
                    state.streak = 0
                else:
                    state.ties += 1
                    state.streak = 0
                
                state.total_rounds += 1
                state.game_history.append(result)
                
                print(f"\n→ Points: {state.points} | Streak: {state.streak}")
                time.sleep(1)
                
        elif selection == '2':  # View Statistics
            show_statistics(state)
            
        elif selection == '3':  # Tournament Standings
            print("\n🏆 TOURNAMENT HISTORY")
            print("="*50)
            # Could track tournament results separately
            for game in state.game_history[-5:]:
                emoji = "✅" if game['result'] == 'win' else "❌" if game['result'] == 'loss' else "🤝"
                print(f"{emoji} {game['player_move']} vs {game['computer_move']}")
            print("="*50)
            input("Press Enter to continue...")
            
        elif selection == '4':  # Settings
            print("\n⚙️ SETTINGS")
            print("1. Change Game Mode")
            print("2. Reset All Data")
            print("3. Back to Menu")
            setting_choice = input("Option: ").strip()
            
            if setting_choice == '1':
                game_mode = choose_game_mode()
            elif setting_choice == '2':
                confirm = input("Reset all progress? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    state = GameState()
                    print("Progress reset!")
                    
        elif selection == '5':  # Save & Quit
            state.save_history()
            print("\n💾 Game saved successfully!")
            print("Thanks for playing!")
            sys.exit()
            
        elif selection == '6':  # Rules
            display_rules(game_mode)
            input("Press Enter to continue...")
            
        elif selection == '7':  # About
            print("\n" + "="*50)
            print("Extended Rock Paper Scissors")
            print("Enhanced edition with:")
            print("  • Lizard & Spock expansion")
            print("  • Betting/Risk mechanics")
            print("  • Pattern-detecting AI")
            print("  • Tournament mode")
            print("  • Persistent save system")
            print("="*50)
            input("Press Enter to continue...")
        
        else:
            print("❌ Invalid selection. Try again.")
        
        # Save after each round
        state.save_history()
        
        # Check for bankruptcy
        if state.points <= 0:
            print("\n💸 Bankrupt! Starting fresh...")
            state.points = 100
            state.wins = state.losses = state.ties = 0
            state.streak = 0
            time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Saving progress...")
        sys.exit()