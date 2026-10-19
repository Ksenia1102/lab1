import pytest

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (
    count_one_bits,
    count_vowels,
    has_unique_characters,
    is_balanced_number,
    mse,
    multiplicative_persistence,
    prime_factorization,
    pyramid,
)


#  count_vowels 

def test_count_vowels_example():
    # Проверяем подсчет гласных в стандартной строке из примера
    assert count_vowels(TextInput("hello world")) == 3


def test_count_vowels_mixed_case():
    # Проверяем, что функция корректно обрабатывает буквы разного регистра
    assert count_vowels(TextInput("HeLLo WoRLD")) == 3


def test_count_vowels_no_vowels():
    # Проверяем строку, в которой полностью отсутствуют гласные буквы
    assert count_vowels(TextInput("xyz")) == 0


def test_count_vowels_empty_string():
    # Проверяем поведение функции при передаче пустой строки
    assert count_vowels(TextInput("")) == 0


def test_count_vowels_all_vowels():
    # Проверяем строку, состоящую исключительно из гласных букв в разных регистрах
    assert count_vowels(TextInput("aeiouAEIOU")) == 10


#  has_unique_characters 

def test_has_unique_characters_true():
    # Проверяем строку, состоящую только из уникальных символов
    assert has_unique_characters(TextInput("abcdef")) is True


def test_has_unique_characters_false():
    # Проверяем строку, содержащую повторяющиеся символы
    assert has_unique_characters(TextInput("hello")) is False


def test_has_unique_characters_empty():
    # Проверяем пустую строку на наличие уникальных символов
    assert has_unique_characters(TextInput("")) is True


def test_has_unique_characters_single_char():
    # Проверяем строку, состоящую из одного символа
    assert has_unique_characters(TextInput("a")) is True


#  count_one_bits 

def test_count_one_bits_example():
    # Проверяем количество единичных битов для числа 7
    assert count_one_bits(PositiveIntegerInput(7)) == 3  # 111


def test_count_one_bits_one():
    # Проверяем количество единичных битов для числа 1
    assert count_one_bits(PositiveIntegerInput(1)) == 1


def test_count_one_bits_power_of_two():
    # Проверяем количество единичных битов для точной степени двойки (число 16)
    assert count_one_bits(PositiveIntegerInput(16)) == 1  # 10000


def test_count_one_bits_zero_bits_except_itself():
    # Проверяем число 255, состоящее сплошь из единичных битов в байте
    assert count_one_bits(PositiveIntegerInput(255)) == 8  # 11111111


#  multiplicative_persistence 

def test_multiplicative_persistence_39():
    # Проверяем мультипликативную устойчивость для числа 39
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3


def test_multiplicative_persistence_single_digit():
    # Проверяем однозначное число 4, для которого шагов не требуется
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0


def test_multiplicative_persistence_999():
    # Проверяем мультипликативную устойчивость для числа 999
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4


def test_multiplicative_persistence_with_zero_digit():
    # Проверяем число 10, содержащее нуль
    # 10 -> 1*0 = 0, однозначное число сразу
    assert multiplicative_persistence(PositiveIntegerInput(10)) == 1


#  mse 

def test_mse_example():
    # Проверяем среднеквадратичную ошибку для одинаковых векторов (должна быть равна нулю)
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0


def test_mse_basic():
    # Проверяем вычисление MSE для различных векторов
    result = mse(VectorPairInput([1, 2, 3], [2, 2, 2]))
    assert result == pytest.approx((1 + 0 + 1) / 3)


def test_mse_single_element():
    # Проверяем расчет MSE для векторов длины 1
    assert mse(VectorPairInput([5], [2])) == pytest.approx(9.0)


#  prime_factorization 

def test_prime_factorization_example():
    # Проверяем разложение на простые множители для числа 86240
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"


def test_prime_factorization_prime_number():
    # Проверяем разложение простого числа 13
    assert prime_factorization(PositiveIntegerInput(13)) == "(13)"


def test_prime_factorization_one():
    # Проверяем граничный случай для числа 1
    assert prime_factorization(PositiveIntegerInput(1)) == ""


def test_prime_factorization_power_of_two():
    # Проверяем разложение степени двойки (число 8)
    assert prime_factorization(PositiveIntegerInput(8)) == "(2**3)"


#  pyramid 

def test_pyramid_possible():
    # Проверяем построение пирамиды из 14 кубиков (высота должна быть равна 3)
    # 1^2 + 2^2 + 3^2 = 14
    assert pyramid(PositiveIntegerInput(14)) == 3


def test_pyramid_single_cube():
    # Проверяем построение пирамиды из одного кубика
    assert pyramid(PositiveIntegerInput(1)) == 1


def test_pyramid_impossible():
    # Проверяем случай, когда построение пирамиды невозможно (число 2)
    assert pyramid(PositiveIntegerInput(2)) == "It is impossible"


#  is_balanced_number 

def test_is_balanced_number_true_even_length():
    # Проверяем сбалансированное число с четным количеством цифр вокруг центра
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True


def test_is_balanced_number_false():
    # Проверяем несбалансированное число
    assert is_balanced_number(PositiveIntegerInput(123456)) is False


def test_is_balanced_number_single_digit():
    # Проверяем однозначное число
    # нет цифр ни слева, ни справа -> суммы равны (0 == 0)
    assert is_balanced_number(PositiveIntegerInput(5)) is True


def test_is_balanced_number_two_digits():
    # Проверяем двузначное число, где средние цифры исключаются из сумм
    # 11 -> средние цифры не входят, обе суммы 0
    assert is_balanced_number(PositiveIntegerInput(11)) is True
