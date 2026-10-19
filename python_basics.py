"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


# Задание 1
def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiouAEIOU")
    return sum(1 for ch in text if ch in vowels)


# Задание 2
def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


# Задание 3
def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


# Задание 4
def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


# Задание 5
def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    total = sum((p - e) ** 2 for p, e in zip(predicted, expected))
    return total / n


# Задание 6
def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = []
    remaining = number
    divisor = 2
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            power = 0
            while remaining % divisor == 0:
                remaining //= divisor
                power += 1
            factors.append((divisor, power))
        divisor += 1
    if remaining > 1:
        factors.append((remaining, 1))

    parts = []
    for base, power in factors:
        if power == 1:
            parts.append(f"({base})")
        else:
            parts.append(f"({base}**{power})")
    return "".join(parts)


# Задание 7
def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k * k
    if total == cube_count:
        return k
    return "It is impossible"


# Задание 8
def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = str(number)
    n = len(digits)
    half = n // 2
    left = digits[:half]
    if n % 2 == 0:
        right = digits[half:]
    else:
        right = digits[half + 1:]
    return sum(int(d) for d in left) == sum(int(d) for d in right)


if __name__ == "__main__":
    print("Ручная проверка\n")
    
    # Задание 1
    v_res = count_vowels(TextInput("hello world"))
    print(f"1. Гласные в 'hello world': {v_res}\n")
    
    # Задание 2
    u_res = has_unique_characters(TextInput("abcdef"))
    print(f"2. Уникальные символы в 'abcdef': {u_res}\n")
    
    # Задание 3
    bits_res = count_one_bits(PositiveIntegerInput(7))
    print(f"3. Единичные биты для 7: {bits_res}\n")
    
    # Задание 4
    mp_res = multiplicative_persistence(PositiveIntegerInput(39))
    print(f"4. Мультипликативная устойчивость 39: {mp_res}\n")
    
    # Задание 5
    mse_res = mse(VectorPairInput([1, 2, 3], [2, 2, 2]))
    print(f"5. MSE [1,2,3] и [2,2,2]: {mse_res}\n")
    
    # Задание 6
    pf_res = prime_factorization(PositiveIntegerInput(86240))
    print(f"6. Разложение 86240: {pf_res}\n")
    
    # Задание 7
    pyr_res = pyramid(PositiveIntegerInput(14))
    print(f"7. Пирамида для 14 кубиков: {pyr_res}\n")
    
    # Задание 8
    bal_res = is_balanced_number(PositiveIntegerInput(1234006))
    print(f"8. Сбалансированность 1234006: {bal_res}")