# ========== STEP 1-2: Build Letter to Points Dictionary ==========

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
           "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
points = [1, 2, 2, 2, 1, 3, 3, 3, 1, 4, 3, 1, 2, 3, 1, 3, 5, 1, 1, 1, 2, 3, 3, 4, 3, 5]

# Using dictionary comprehension with zip() to map letters to points
letter_to_points = {letter: point for letter, point in zip(letters, points)}

# Confirm structure (commented out for clean output)
# print(letter_to_points)


# ========== STEP 3-8: Score a Word ==========

def score_word(word):
    """Calculate the total points for a given word."""
    point_total = 0
    
    for letter in word:
        # Add point value, or 0 if letter not found
        point_total += letter_to_points.get(letter, 0)
    
    return point_total

# Test with BROWNIE (should be 12 points)
brownie_points = score_word('BROWNIE')

print(f"Brownie points: {brownie_points}")  # Expected: 12


# ========== STEP 9-14: Score a Game ==========

# Track which words each player has played
player_to_words = {
    "wordNerd":  ["BLUE", "EARTH", "ERASER", "ZAP", "EYES", "BELLY", "COMA"],
    "Lexi Con":  ["EARTH", "ERASER", "ZAP", "EYES", "BELLY", "COMA"],
    "Prof Reader": ["ZAP", "EYES", "BELLY", "COMA"]
}

# Alternative representation matching original data structure
player_to_words_original = {
    "player1":   ["BLUE", "EARTH", "ERASER", "ZAP"],
    "wordNerd":  ["BLUE", "EARTH", "ERASER", "ZAP", "EYES", "BELLY", "COMA", "EXIT", "MACHINE", "HUSKY", "PERIOD"],
    "Lexi Con":  ["EARTH", "ERASER", "ZAP", "EYES", "BELLY", "COMA", "MACHINE"],
    "Prof Reader": ["ZAP", "EYES", "BELLY", "COMA"]
}

# Track total points for each player
player_to_points = {}

for player, words in player_to_words.items():
    player_points = 0
    for word in words:
        player_points += score_word(word)
    player_to_points[player] = player_points

print("\n=== Current Standings ===")
for player, points in player_to_points.items():
    print(f"{player}: {points} points")


# ========== STEP 15 EXTENSIONS ==========

def play_word(player, word):
    """Add a word to a player's word list."""
    if player not in player_to_words:
        player_to_words[player] = []
    player_to_words[player].append(word)
    
    # Auto-update points after adding word
    update_point_totals()
    print(f"{player} played '{word}'!")

def update_point_totals():
    """Recalculate all player points based on current word lists."""
    global player_to_points
    player_to_points = {}
    
    for player, words in player_to_words.items():
        player_points = sum(score_word(word) for word in words)
        player_to_points[player] = player_points

# Make letter_to_points handle lowercase (case-insensitive scoring)
letter_to_points_lower = {letter.lower(): point for letter, point in zip(letters, points)}

def score_word_v2(word):
    """Enhanced scoring function that handles lowercase input."""
    return sum(letter_to_points_lower.get(letter.lower(), 0) for letter in word)

def score_word_final(word):
    """Final version - handles both upper and lowercase, falls back to original dict."""
    word_upper = word.upper()
    return sum(letter_to_points.get(letter, 0) for letter in word_upper)


# ========== DEMONSTRATION ==========

print("\n=== Testing Case Insensitivity ===")
print(f"BROWNIE: {score_word_final('BROWNIE')}")
print(f"brownie: {score_word_final('brownie')}")
print(f"BrOnIe: {score_word_final('BrOnIe')}")

print("\n=== Playing New Words ===")
play_word("wordNerd", "PYTHON")
play_word("Lexi Con", "DICTIONARY")
update_point_totals()

print("\n=== Final Leaderboard ===")
sorted_players = sorted(player_to_points.items(), key=lambda x: x[1], reverse=True)
for player, points in sorted_players:
    print(f"{player}: {points} points")


# ========== VISUALIZATION OF LETTER POINTS ==========