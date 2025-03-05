import numpy as np

# TODO [1]: implement the guessing_game function
# hint: return type is tuple[bool, list[int], int]:
def guessing_game(max: int, *, attempts: int) ->  tuple[bool, list[int], int]:
    
    # Generating a random number
    generated_number: int = np.random.randint(1, max+1)
    guesses: list[int] = []
    state: bool = False

    while attempts > 0 and state == False:
        # Handling a single attempt
        user_guess: int = None

        while user_guess is None:
            try:
                user_guess = int(input("Enter your guess: "))
            except ValueError:
                print("Your guess should be an integer number.")
            else:
                guesses.append(user_guess)
                attempts-=1

                if user_guess == generated_number:
                    print("Correct, remaining attempts: ", attempts)
                    state = True
                    break
                elif user_guess > generated_number:
                    print("Too High, remaining attempts: ", attempts)
                elif user_guess < generated_number:
                    print("Too Low, remaining attempts: ", attempts)
    
    return (state, guesses, generated_number)


# TODO [2]: implement the play_game function
def play_game()-> None:
    max_value:int = 20
    attempts:int = 5
    
    (state, guesses, generated_number) = guessing_game(max_value, attempts = attempts)  

    assert (state != (generated_number not in guesses)) , "Failed!!"
          

