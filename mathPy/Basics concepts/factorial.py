def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

n = int(input("Enter a number: "))
print(f'The factorial of this number is: [{factorial(n)}]')
print("Ending program...")

""" 
^^ Other ways to do the same:

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# ... existing code ...
n = int(input("Digite um número: "))
print(f'O fatorial deste número é: [{factorial(n)}]')
print("Encerrando o programa...")

import math

# ... existing code ... 
n = int(input("Digite um número: "))
print(f'O fatorial deste número é: [{math.factorial(n)}]')
print("Encerrando o programa...") 

"""

print()