import numpy as np

# TODO [1]: implement the guessing_game function
def guessing_game(max: int, attempts: int) : # hint: return type is tuple[bool, list[int], int]:
    value = np.random.randint(1, max + 1)
    while attempts > 0:
        guess = None
        while guess is None:
            try:
                guess = int(input("Guess: "))
            except:
                print("Enta sus")
        if guess < value:
            print("Too low")
            attempts -=1
        elif guess > value:
            print("Too high")
            attempts -=1
        else:
            print("Correct")
            break
    

# TODO [2]: implement the play_game function
def play_game()-> None:
    max_value:int = 20
    attempts:int = 5
    guessing_game(max_value, attempts)
            

