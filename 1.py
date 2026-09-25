my_set = {1, 2, 3, 2, 1}
my_set.add(4)
my_set.discard(1)
print(my_set)
def factorial(n):
    if n == 0:          # Base case
        return 1
    else:
        return n * factorial(n - 1)   # Recursive case
print(factorial(5)) 