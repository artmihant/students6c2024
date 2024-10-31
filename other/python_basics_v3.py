# Это краткое пособие по языку Python написано на корректном коде языка Python версии 3.6
# Многочисленные комментарии обозначаются символом #.


# Настоящий документ является вольным дополненным переводом справки по синтаксису языка программирования, доступной в оригинале по адресу 
# http://cheat.sh/python/:learn .
# Пользуясь случаем, рекомендую этот ресурс для ознакомления с новыми языками и инструментами.


# Для удобной работы с пособием используйте любой редактор кода с подсветкой.  
# В качестве редактора кода с подсветкой рекомендую notepad++ или sublime text.


""" Многострочные строки можно писать, 
    используя три символа кавычки ", 
    что часто используется как многострочный комментарий.
"""


####################################################
# 0. Установка
####################################################

# Для работы с Python вам потребуется установленный в вашей операционной системе интерпретатор языка программирования Python.

# Вы можете скачать его с https://www.python.org/downloads/ или получить из репозитория

# Вы можете проверить, установилась ли программа, открыв системный терминал (для windows - powershell) и набрав python.

# Таким образом, вы запустите интерпретатор в интерактивном режиме (REPL). В этом режиме вы можете заниматься отладкой вашего кода, проврять синтаксис или использовать язык как программируемый калькулятор.

# В большинстве случаев вам нужно создать файл с расширением .py с текстом программы внутри. 

# Попробуйте создать файл hello.py с единственной следующей строкой:

print('Hello, world!')

# и напечатать в терминале из той директории, в которой вы создали файл

# >>> python hello.py

# Должно вывести в терминал "Hello, world!"

# Для математиков будет хорошей идеей установить Jupiter Lab и работать из неё (не разбирается в данном пособии)



####################################################
# 1. Простейшие типы данных и операторы
####################################################

# У вас есть два основных числовых типа  - int и float
3  # => 3 # целое число (int).
3.2 # => 3.2 # число с плавающей запятой (float).

type(3) # => <class 'int'>
type(3.0) # => <class 'float'>


# Математика такая же, как и ожидается
1 + 1  # => 2
8 - 1  # => 7
10 * 2  # => 20
5 / 2  # => 2.5

# При делении всегда результат - float. Однако при сравнении, 2 == 2.0 => True 
35 / 5  # => 7.0

# Результат целочисленного деления усекается как для положительных так и отрицательных чисел.
5 // 3  # => 1
5.0 // 3.0  # => 1.0 # Работает и для float
-5 // 3  # => -2
-5.0 // 3.0  # => -2.0

# Операция взятия остатка
7 % 3  # => 1

# Возведение в степень
2 ** 4  # => 16

# Числовой тип int изначально поддерживает длинную арифметику
2 ** 256  # => 115792089237316195423570985008687907853269984665640564039457584007913129639936

# В отличие от float
2.0 ** 256 # => 1.157920892373162e+77

# Круглые скобки работают, как вы и ожидаете
(1 + 3) * 2  # => 8

# Булевы операции 
# Весь синтаксис, все операторы, имена переменных и т.п. в python чувствительны к регистру.
True and False  # => False
False or True  # => True

# Булевы операторы можно использовать с целыми числами
0 and 2  # => 0
-5 or 0  # => -5
0 == False  # => True
2 == True  # => False
1 != True  # => False

# Отрицание not
not True  # => False
not False  # => True

# Равенство ==
1 == 1  # => True
2 == 1  # => False

# Неравенство !=
1 != 1  # => False
2 != 1  # => True

# Разные сравнения
1 < 10  # => True
1 > 10  # => False
2 <= 2  # => True
2 >= 2  # => True

# Сравнения можно выстроить цепью!
1 < 2 < 3  # => True
2 < 3 < 2  # => False
2 == 2 != 1 # => True

# Строки обозначаются кавычками " или ' , без разницы.
"This is a string."
'This is also a string.'

type('hello') # => <class 'str'>

# Строки можно складывать.
"Hello " + "world!"  # => "Hello world!"
# Знак '+' не обязателен
"Hello " "world!"  # => "Hello world!"

# ... или умножать
"Hello" * 3  # => "HelloHelloHello"

# Можно обратиться к отдельному символу, в том числе и кириллическому. По умолчанию в python3 все строки кодированы UTF-8
"This is a string"[0]  # => 'T'

"Это строка"[0] # => 'Э' 

# Так можно узнать длину строки или любого другого списка
len("This is a string")  # => 16

# Строки можно форматировать с помощью % . В этом примере x и y обязаны быть строками. 
# Буква соответствует спецификатору вывода, аналогично тем, что в printf в Си.
x = 'apple'
y = 'lemon'
z = "The items in the basket are %s and %s" % (x, y)

# Гораздо удобнее использовать метод format -
#   он автоматически приводит все значения к строковому типу.
"{} is a {}".format("This", "placeholder")
"{0} can be {1}".format("strings", "formatted")

# Можно использовать ключевые слова для простоты орентировки.
"{name} wants to eat {food}".format(name="Bob", food="lasagna")

# В python 3.6 добавили f-строки . Вот пример их применения:
name="Bob"
food="lasagna"
f"{name} wants to eat {food}" # Bob wants to eat lasagna

# format и f-строки автоматически приводит нестроковые объекты к строковому представлению,
# (если это возможно)
count = 5
f"Вася съел {count} тортиков"  # Вася съел 5 тортиков

# На текущий момент f-строки - лучший метод из всех, используйте его. 

# Помимо буквы f перед строкой можно вписать букву r
r"\n" # с буквой r будет читаться как \n
"\n" # без неё выйдет перенос строк. Есть и другие спецсимволы.
"\\n" # Во так \ экранируется. Строки, содержащие \, без r будут отформатированы.
# Так же есть b, обозначающая байтовую строку.

# None это пустое значение
None  # => None

# Не используйте "==" для сравнения обьекта с None
# Используете оператор "is"
"" is None  # => False
None is None  # => True

# Оператор 'is' проверяет идентичность объектов,
#   то есть равенство адресов в памяти, на которые они ссылаются. 
# Оператор is можно применять для сравнения с None, True, False и проверки тождества обьектов. 
# Оператор is не стоит применять для сравнения строк или чисел.

# Объектом в питоне считается абсолютно всё, что можно поместить в переменную. Числа и строки тоже объекты.

# Любой обьект может быть задействован в булевом контексте.
# Вот что интерпретируется как False:
#    - None
#    - ноль любого числового типа (т.к., 0, 0L, 0.0, 0j)
#    - пустая последовательность (т.к., '', (), [])
#    - пустой контейнер (т.к., {}, set())
#    - экземпляры пользовательских классов в специальных условиях, определяемых разработчиком класса
#
# Любые другие значения считаются истиными (используете функцию типа bool() на них для определения).

bool(0)  # => False
bool("")  # => False
bool("False")  # => True


####################################################
# 2. Переменные и коллекции
####################################################

# В python есть оператор вывода (в текущий поток вывода; обычно это консоль, где исполняется скрипт или файл лога).
# Он выведет всё, что вы в него введете, через пробел.
print("Nice to meet you!", 42)  # => Nice to meet you! 42

# Простей способ считать данные из консоли
user_input = input("Enter some data: ")  # Вернет ввод как питоновский код
# теперь в user_input то, что напечатано, в виде строки. 

# Очень часто в учебных задачах нужно принять из консоли число. Это делается так:
num1 = int(input("Введите целое число: "))
num2 = float(input("Введите дробное число: "))

# Любое имя типа: int, float, bool, str может быть использовано как функция приведения типов.

# Переменные не нужно декларировать перед их назначением.
some_var = 5
# Используете строчные_буквы_с_подчеркиваниями для переменных - это считается аккуратным кодом.
some_var  # => 5

# Если вы захотите считать неназначенную переменную, это создаст исключение.
# См. Поток управления, чтобы узнать больше об обработке исключений.
some_other_var  # Вызовет NameError: name 'some_other_var' is not defined

# if можно использовать таким образом
# Эквивалентно тернарному оператору '?:' в Си
some_var = "yahoo!" if 3 > 2 else "ou, no"  # => "yahoo!"

# Список (list) это последовательность произвольных значений
li = []
# Можно сразу задать оперделенный список
other_li = [4, 5, 6]

# Добавить элемент в конец списка можно с помощью append
li.append(1)  # li теперь [1]
li.append(2)  # li теперь [1, 2]
li.append(4)  # li теперь [1, 2, 4]
li.append(3)  # li теперь [1, 2, 4, 3]
# Удалить элемент из конца списка и вернуть его
li.pop()  # => 3 and li is now [1, 2, 4]
# Вернуть обратно
li.append(3)  # li is now [1, 2, 4, 3] again.

# К элементам списка возможен доступ по индексу
li[0]  # => 1
# Можно переназначать значения элементам списка =
li[0] = 42
li[0]  # => 42

# Получить последний элемент
li[-1]  # => 3

# Попытка получть элемент за пределами списка поднимает исключение IndexError
li[4]  # Поднимает IndexError

# Можно получить срез списка.
li[1:3]  # => [2, 4]
# Отсчитывая с начала 
li[2:]  # => [4, 3]
# Не доходя до конца
li[:3]  # => [1, 2, 4]
# Выбрать каждый второй
li[::2]  # =>[1, 4]
# Получить перевернутую копию списка
li[::-1]  # => [3, 4, 2, 1]
# Комбинируйте для получения нужного вам среза
# li[start:end:step]

# Удалить элемент из списка можно оператором "del"
del li[2]  # li теперь [1, 2, 3]

# Вы можете складывать списки 
li + other_li  # => [1, 2, 3, 4, 5, 6]
# При этом списки li и other_li не изменятся.

# Конкантинация списков с помощью "extend()"
li.extend(other_li)  # li теперь [1, 2, 3, 4, 5, 6]

# Удалить первое включение элемента "2"
li.remove(2)  # li теперь [1, 3, 4, 5, 6]
li.remove(2)  # Поднимет ValueError если 2 не в списке

# Вставить элемент на оперделенное место
li.insert(1, 2)  # li снова [1, 2, 3, 4, 5, 6] 

# Получить индекс первого найденного элемента
li.index(2)  # => 1
li.index(7)  # Исключение ValueError: 7 вне границ списка

# Проверить наличие элемента в списке с помощью "in"
1 in li  # => True

# Узнать длину списка "len()"
len(li)  # => 6


# Существует особый синтаксичесйи сахар для генерации списков
values = [-x for x in [1, 2, 3, 4, 5]]
for x in values:
    print(x)  # напечатает -1 -2 -3 -4 -5 с переносом между значениями.

# Можно сразу перегнать генерируемые значения в список
values = [-x for x in range(1,6)]
print(values)  # => [-1, -2, -3, -4, -5]


# Кортедж (tuple) похож на список, но он неизменяем.
# Используете списки для данных, чья длина заранее неопределена 
#   а кортеджи - для коротких структурованных данных. 
# Например, вектор из трех координат хорошо выглядит в кортедже. 

tup = (1, 2, 3)
tup[0]  # => 1
tup[0] = 3  # Поднимет TypeError

# Многие операции, применимые для списков, работают и здесь
len(tup)  # => 3
tup + (4, 5, 6)  # => (1, 2, 3, 4, 5, 6)
tup[:2]  # => (1, 2)
2 in tup  # => True

# Можно распаковать кортедж (или список) в переменные
a, b, c = (1, 2, 3)  # теперь a = 1, b = 2, c = 3
d, e, f = 4, 5, 6  # даже не обязательно ставить скобки
# Кортеджи можно создавать, не указывая скобки
g = 4, 5, 6  # => (4, 5, 6)
# Только посмотрите, как легко теперь поменять местами значения двух переменных
e, d = d, e  # теперь d = 5 и e = 4

# Словарь (dict) хранит пары ключ - значение
empty_dict = {}
# Его можно создать и вот так
filled_dict = {"one": 1, "two": 2, "three": 3}

# Доступ к значению можно получить с помощью []
filled_dict["one"]  # => 1

# Вот так можно получить список ключей
filled_dict.keys()  # => ["one", "three", "two"]
# Внимание - ключи, строго говоря, не будут по порядку заполнения.
# Ваш результат может быть иным с точностью до перестановки.

# Получить все значения список
filled_dict.values()  # => [1, 3, 2]
# Перестановка значения относительно исходного заполнения будет такая же, как и у ключей.

# Получить список пар ключ-значение 
filled_dict.items()  # => [("one", 1), ("three", 3), ("two", 2)]

# Проверить наличие ключа в словаре
"one" in filled_dict  # => True
1 in filled_dict  # => False

# Попытка запросить несущестующий ключ вызовет KeyError
filled_dict["four"]  # KeyError

# Используйте метод get(), что бы не вызвать KeyError
filled_dict.get("one")  # => 1
filled_dict.get("four")  # => None
# Можно подставить в get аргумент, который вернется есfilled_dictли не найдется ключ
filled_dict.get("one", 4)  # => 1
filled_dict.get("four", 4)  # => 4

# Создать новое или отредактировать старое значение по ключу очень просто
filled_dict["four"] = 4  # now, filled_dict["four"] => 4

# "setdefault()" вставит значение, только если ключ отсутствовал
filled_dict.setdefault("five", 5)  # filled_dict["five"] установлен в 5
filled_dict.setdefault("five", 6)  # filled_dict["five"] всё ещё 5

# Последняя из важных структур данных - множества (set).
# множества изменяемы, но в них нет дубликатов и нет сохранения порядка следования элементов.

empty_set = set()
# Можно создать множество из списка для получения уникальных значений
some_set = set([1, 2, 2, 3, 4])  # some_set = {1, 2, 3, 4}

# Порядок не гарантируется, система попробует отсортировать элементы
another_set = set([4, 3, 2, 2, 1])  # another_set = {1, 2, 3, 4]}

# Но у нее может не совсем получиться
diverse_set = {1,2,3,'a','b','c'} # diverse_set = {1, 2, 3, 'b', 'c', 'a'}

# Скобки {} можно использовать для создания множества
filled_set = {1, 2, 3, 4}  # => {1, 2, 3, 4}

# Добавление элеменов во множество
filled_set.add(5)  # filled_set = {1, 2, 3, 4, 5}

# Главная магия множеств - на них можно использовать хорошо знакомые вам операции теории множеств
A = {1, 2, 3, 4, 5}
B = {3, 4, 5, 6}

# Объединение &
A & B  # => {3, 4, 5}

# Пересечение |
A | B  # => {1, 2, 3, 4, 5, 6}

# Разность -
{1, 2, 3, 4} - {2, 3, 5}  # => {1, 4}

# Симметрическая разность ^
{1, 2, 3, 4} ^ {2, 3, 5}  # => {1, 4, 5}

# Проверка на включение
{1, 2} >= {1, 2, 3}  # => False

# Множество слева - подмножество множества справа
{1, 2} <= {1, 2, 3}  # => True

# Проверка на наличие элемента во множестве
2 in filled_set  # => True
10 in filled_set  # => False


####################################################
#  3. Поток управления
####################################################

# Создадим-ка переменную. Задавать тип при объевлении не нужно.
var = 5

# Вот так задается if. Отступы важны в python!

if var > 10: 
    var -= 1
    print(f"var больше 10. Уменьшим! Теперь он {var}.")
elif var < 10: 
    var += 1
    print(f"var меньше 10. Увеличим! Теперь он {var}.")
else: 
    print("var равен 10. Идеально!")

# напечатает "var меньше 10. Увеличим! Теперь он 6."

# Установите за правило использовать определенный тип отсупа во всех своих проектах. 
# В этом пособии везде будет отступ в четыре пробела. Это хорошая практика, но мы не настаиваем.
# Если интерпретатор обнаружит разные типы отступов в одном файле с кодом, скорее всего он выдаст ошибку.

# Как правило, код в python форматируется посредством отступов. 
# Удобство этого похода может быть сперва неочевидно, но по мере практики становится очень ясным
# Однако, если вам очень хочется, вы можете переписать код выше так:

var = 5
if var > 10: var -= 1; print(f"var больше 10. Уменьшим! Теперь он {var}")
elif var < 10: var += 1; print(f"var меньше 10. Увеличим! Теперь он {var}")
else: print("var равен 10. Идеально!")

# Такой подход не приветствуется стандартом языка. 
# Используете ";"" , только если вам действительно нужно вытянуть код в одну строку, например при передаче команды. 

"""
Цикл for проходит по списку
напечатает:
    Собака это животное
    Кошка это животное
    Мышка это животное
"""
for animal in ["Собака", "Кошка", "Мышка"]:
    print("{animal} это животное")

"""
"range(number)" вернет список целых чисел от нуля до number
напечатает:
    0
    1
    2
    3
"""
for i in range(4):
    print(i)

"""
"range(lower, upper)" вернет числа от lower до upper, последовательно, не включая upper
напечатает:
    4
    5
    6
    7
"""
for i in range(4, 8):
    print(i)

"""
Цикл while исполняется, пока условие внутри цикла истинно.
напечатает:
    0
    1
    2
    3
"""
x = 0
while x < 4:
    print(x)
    x += 1  # То же, что и x = x + 1


# Вот так можно открыть текстовый файл на чтение (на запись файлы открывают с флагом 'w').
# Конструкция with - as не забудет его закрыть, и его дескриптор не будет висеть, блокируя файл на уровне OC.

with open("myfile.txt", 'r') as f:
    f.read()
    for line in f:
        print(line)

# Это полностью аналогично такому коду:

f = open("myfile.txt", 'r')
for line in f:
    print(line)
f.close()

# но последнем случае вам самим нужно позаботиться о закрытии файлового дескриптора. 
# В прочем, по завершению скрипта все незакрытые дескрипторы закроются автоматически. 


####################################################
# 4. Функции
####################################################

# Используйте def для создания новой функции
def add(x, y):
    print(f"x is {x} and y is {y}")
    return x + y  # Вернет значение после return 

# Вызов функции с параметрами
add(5, 6)  # => выведет "x is 5 and y is 6" и вернет 11

# Другое способ вызвать функцию - использовать названия аргументов как ключевые слова
add(y=6, x=5)  # Ключевые слова могут быть переданы в любом порядке.


# Можно создать функцию, принимающую произвольное число аргуметов
# теперь внутри функции, args - коредж с передаными аргументами
def varargs(*args):
    return args


varargs(1, 2, 3)  # => (1, 2, 3)


# Если использовать **, в kwargs передастся не кортедж а словарь
# ключи - ключевые слова, переданные в функцию
def keyword_args(**kwargs):
    return kwargs


# Посмотрим, что получилось
keyword_args(big="foot", loch="ness")  # => {"big": "foot", "loch": "ness"}


# Можно использовать и то и другое, если хочется
def all_the_args(*args, **kwargs):
    print(args)
    print(kwargs)

all_the_args(1, 2, a=3, b=4)
""" выведет:
    (1, 2)
    {"a": 3, "b": 4}
"""

# Распаковка кортеджа или словаря может происходить и с другой стороны
args = (1, 2, 3, 4)
kwargs = {"a": 3, "b": 4}
all_the_args(*args)  # то же, что и  all_the_args(1, 2, 3, 4)
all_the_args(**kwargs)  # то же, что и all_the_args(a=3, b=4)
all_the_args(*args, **kwargs)  # то же, что и all_the_args(1, 2, 3, 4, a=3, b=4)


# можно передавать всю информацию, переданную в функцию, по цепочке
def pass_all_the_args(*args, **kwargs):
    all_the_args(*args, **kwargs)
    print(varargs(*args))
    print(keyword_args(**kwargs))

# Область действия функции 

x = 5
def set_x(num):
    # Локальная переменная x не связана с глобальной
    x = num
    print(x)  # => 43

def set_global_x(num):
    global x
    print(x)  # => 5
    x = num  # теперь глобальная переменная x = 6
    print(x)  # => 6


print(x)  # => 5
set_x(43)
print(x)  # => 5
set_global_x(6)
print(x)  # => 6


# Функция в питоне - обьект первого типа, т.е. её можно передать по ссылке.
def create_adder(x):
    def adder(y):
        return x + y
    return adder


add_10 = create_adder(10)
add_10(3)  # => 13

# Можно создать анонимную функцию
(lambda x: x > 2)(3)  # => True
(lambda x, y: x ** 2 + y ** 2)(2, 1)  # => 5

# Есть встроеные функции высшего порядка
map(add_10, [1, 2, 3])  # => [11, 12, 13]
map(max, [1, 2, 3], [4, 2, 1])  # => [4, 2, 3]

filter(lambda x: x > 5, [3, 4, 5, 6, 7])  # => [6, 7]

# Можно использовать генераторы списков с тем же эффектом, что и map и filter
[add_10(i) for i in [1, 2, 3]]  # => [11, 12, 13]
[x for x in [3, 4, 5, 6, 7] if x > 5]  # => [6, 7]

# Точно так же можно генерировать множества и словари.
{x for x in 'abcddeef' if x in 'abc'}  # => {'a', 'b', 'c'}
{x: x ** 2 for x in range(5)}  # => {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}



####################################################
#  5. Изменяемые и неизменяемые типы данных
####################################################

# При передаче в функцию значения можно столкнуться со следующей ситуацией

def append_three(x):
    x.append()

some_list = [1,2]

append_three(some_list)

some_list # [1,2,3]

# В этом примере функция append_two неявным образом изменила переданный в неё список. Это может быть как желаемым, так и не желаемым поведением. Дело в том, что list является изменяемым типом. Знакомые с Си программисты могут считать, что list всегда передается в функцию по ссылке. 
# Другие типы, например, tuple, неизменяемы:

some_tuple = (1,2)

append_three(some_list) # выдаст ошибку: у tuple нет метода append. Но как бы не была написана функция append_three , в данном случае у программиста нет возможности как-либо изменить изнутри неё состояние переменной some_tuple. 

# Неизменяемыми являеются следующие типы: int, float, str, bool, tuple. Все остальные типы, включая классы, являются изменяемыми.



####################################################
# 6. Классы
####################################################

"""
Обьектно-ореентированное программирование (ООП) - весьма мощная парадигма. На определенном уровне своего развития программист понимает, что львиная доля его профессиональной деятельности сводится к процессу борьбы со сложностью программного кода. ООП позволяет частично обуздать эту сложность.
Суть в том, что вы описываете сущностно связанные данные и основную логику работы с ним в специальной форме, называемой классом. Затем вы создаете экземпляры класса - объекты, помещая в обьект данные, характерные для него. После этого, всё, что вам остается делать сводится к манипуляции объектами.
Терминология ООП:
* Класс - описание на языке программирования того, какие данные у нас есть и что с ними можно сделать. В python, типы и классы отличаются в основном тем, что типами называются изначально встроенные конструкции, а классами - привнесенные.
* Объект - экземпляр класса с заполнеными данными. 
* Состояние - соввокупность хранящейся внутри обьекта информации, его индивидуальность
* Атрибут - переменная внутри обьекта, его свойство. Задает часть состояния обьекта. 
* Метод - функция внутри объекта, как правило, изменяющая его или возвращающая какие-то данные о нем. Обычно аппелирует  внутри себя к атрибутам и/или меняет через них состояние обьекта
* Наследование - механизм, позволяющий классам наследоваться друг-от-друга. Один из ключевых механизмов ООП. 

Идея в том, что при разработке сложной системы (например, компьютерной игры), мы определяем базовый класс, затем подклассы, затем подклассы подкласов.

    Цепочка классов может быть, например, такой:
    "сущность, имеющая спрайт на поле" - к этому классу относятся герой, лошадь героя монстры, принцесса, деревья, скалы, плиты дороги... - класс имеет ссылку на файл спрайта и логику его геометрического размещения, возможно параметры проходимости героя сквозь него и.т.п.
    -> "НПС" - мы добавляем логику способности обьекта "жить", "умирать", "двигаться", и т.п. При этом часть логики можеть не прописана, например мы можем не знать, будет ли данный НПС способен двигаться (может это волшебное дерево?) или умереть.
    -> "монстр" - мы добавляем логику поведения при встрече с героем, атаки, возможно хп
    -> "гоблин" - мы добавляем какие-то свойства, присущие только гоблинам. Например, мы хотим, что бы гоблины при встрече с игроками ругались на гоблинском языке, мы прописываем сюда логику этой ругани. 
    -> "зеленый гоблин" - а вот это уже конкретная разновидность монстра, с жестко зафиксироваными HP, спрайтами на все случаи его гоблинской жизни, характером атаки и особеннотями поведения.

    Выстраивая правильное наследование, мы не дублируем кучу однотипного кода, описывающего одинаковую логику, например, отображения спрайтовов у разных игровых сущностей. При этом если мы хотим добавить, условно, "волшебной фее" особенный спрайт с блёстками, мы легко можем переписать именно у фей базовый спрайт. 
    Характерно, что первые три-четыре класса в цепочке: "отображаемая сущность" -> "НПС" -> "монстр" -> "гоблин" -> "зеленый гоблин" не могут породить обьект сами по себе - не существует "просто монстра", и какая-то часть логики, варьирующаясяя от монстра к монстру, может быть не прописана Такие классы зовутся "абстрактными".

* Интерфейс - набор методов, через которые предпологается взаимодействовать с обьектом извне.

* Инкапсуляция - механизм, позволяющий запретить использовать некоторые методы и считывать некоторые атрибуты, вызывая их у обьекта явно. 
    Полезно, если у объекта есть продуманный интерфейс, и операции не через него могут привести к непрогнозируемому исходу. 
    В python инкапсулируются все методы и атрибуты, начинающиеся с __двойного_подчеркивания и не заканчивающиеся им.
    Пример из игры с гоблинами. Пусть гоблин имеет флаг "жив/мертв" и HP бар. Мы не можем позволить чему-то извне класса монстра воздействовать непосредственно на флаг жизни, потому что становится возможно ввести его в состояние, когда гоблин "убит", но его HP > 0, а это почти наверняка ошибка. По сходим причинам, мы не можем разрешить чему-то извне напрямую воздействовать на число HP. То, что мы на самом деле можем - это обратиться к методу класса монстра attaked, который принимает на вход информацию об атаке (возможно, тоже в форме обьекта специального класса атак), и обрабатывает то, сколько HP гоблин потеряет и выживет ли он после такой потери. Это инкапсуляция записи, но нередко бывает и инкапсуляция чтения, когда класс содержит скрытую информацию состояния (скрытые атрибуты), которые не требуется другим классам для взаимодействя с ним и которые другим классам не стоит давать теоретическую возможность читать. 

* Полиморфизм - возможность прописать совершенно различным классами (или типам) логику работы через одинаковые интерфейсы. 
    В python реализовано, в частности, через магические методы: __методы_выделенные_парой_подчеркиваний__
    Справка по магическим методам https://habr.com/ru/post/186608/
    Магические методы позволяют реализовать полиморфизм по базовым операциям, 
    так что вы сможете, например, складывать ваши вектора через оператор как-то так: vector_sum = vector1 + vector2

"""

# В качестве примера рассмотрим, как мог бы быть реализован гоблин в комптьютерной игре про сражения с гоблинами. Здесь всё будет примитивнее, чем в тексте выше, рассматривать просто как абстрактный пример:

# По умолчанию классы наследуются от базового типа object. 
class Goblin(object):
    # Атрибут класса. Будет во всех экземплярах класса (порожденных из него обьектах)

    species = "goblin"

    __fury = 0 # пусть у гоблина есть скрытый параметр ярость, влияющий на атаку

    # Инициализация. Вызывается в момент создания обьекта.
    def __init__(self, name, lvl=1):
        """ Это так называемая документация класса. 
        Считается хорошим тоном описать здесь ваш класс для других программистов
        В данном случае у гоблина будет имя, уровень и здоровье. 
        Максимум hp и сила атаки будут вычисляться из уровня и ярости. """
        self.name = name
        self.lvl = lvl
        self.max_hp = self.calc_max_hp(lvl)
        self.hp = self.max_hp

    def say(self, msg):
        """ В качестве первой переменной в методы передается сам обьект """
        return f"{self.name} growls: {msg}, humies"

    @staticmethod
    def calc_max_hp(lvl):
        """ для @staticmethod в первую переменную текущий обьект не передается
            ещё есть декоратор @classmethod, когда в первую переменную передается класс
        """
        return 5*lvl+10

    def enrage(self, extra_rage):
        self.__fury += extra_rage

    @property
    def attack(self):
        """ Вычисляемые свойства позволяют вызвать функцию как переменную """
        return self.lvl + self.__fury + 3 

    @attack.setter
    def attack(self, attack):
        self.__fury = attack - self.lvl - 3

    

# Инициализация класса. Мы создали гоблина по имени Grolg первого уровня
some_goblin = Goblin(name="Grolg")
print(some_goblin.say("hi"))  # напечатает "Grolg growls: hi, humies"

# Узнаем атаку
some_goblin.attack  # => 4

# Злим гоблина:

some_goblin.__fury += 10 # Разозлить не удалось: AttributeError: 'Goblin' object has no attribute '__fury'
some_goblin.enrage(10) # Вот теперь сработало

some_goblin.attack  # => 14

# Накладываем на гоблина заклинание, уменьшающее атаку на 5:

some_goblin.attack -= 5
some_goblin.attack  # => 9 # но на самом деле уменьшился __fury


# Пусть мы хотим, что бы боссы в нашей игре имели свой собственный класс, 
# похожий на стандартный, но c другими формулами расчета атаки и максимума жизни.
# Мы можем создать новый класс из имеющегося, до- или пере- определив часть его свойств. Это и называется "наследование":

class GoblinBoss(Goblin):
    species = "goblin boss"

    __fury = 5

    @staticmethod
    def calc_max_hp(lvl):
        return 10*lvl+30

    @property
    def attack(self):
        return self.lvl + self.__fury + 10

    @attack.setter
    def attack(self, attack):
        self.__fury = attack - self.lvl - 10


goblin_king = GoblinBoss("Ghazghkull III", lvl=10)
print(goblin_king.say("fall down"))  # напечатает "Ghazghkull III growls: fall down, humies"

goblin_king.attack # 25
goblin_king.hp # 130



####################################################
# 7. Модули
####################################################

# Вы можете импортировать модули
import math

print(math.sqrt(16))  # => 4

# Можно импортировать только отдельные функции
from math import ceil, floor

print(ceil(3.7))  # => 4.0
print(floor(3.7))  # => 3.0

# Можно импортировать все фунции из модуля. Но это не рекомендуется.
from math import *

# Можно переименовать модуль при импорте.
import math as m

math.sqrt(16) == m.sqrt(16)  # => True
# Как видите, импортируемые функции эквиваленты.
from math import sqrt

math.sqrt == m.sqrt == sqrt  # => True

# Модули python - обычные файлы с кодом. Вы и сами можете написать такой.
# Имя модуля совпадает с именем файла (без .py) 

# Вы можете узнать, какие функции и атрибуты содержит модуль.
dir(math)


# Если вы имеете скрипт math.py в той же директории, что и ваш вызываемый скрипт, math.py будет подгружен вместо стандартной библиотеки math. 
# Это происходит потому, что приоритет локальной папки выше, чем приоритеты встроенных библиотек. 
# Пути, по которым можно найти коды встроенных библиотек, по порядку приоритетов, можно найти в переменной sys.path 
# (как вы догадываетесь, сперва нужно подгрузить модуль sys) 



####################################################
# 8. Продвинутые функции языка 
####################################################


# Генераторы.

# Генератор "генерирует" значения по мере их запроса. 
# Генераторы и списки могут принимать участие в конструкции for value in values_list в качестве values_list , 
# но генераторы часто делают это с существенной экономией оперативной памяти.

# Пример. Следующая функция (не генератор!) удвоит все значения списка, который в него поместили. 
# Если список достаточно велик, будут проблемы с загрузкой/выгрузкой его из оперативной памяти.
def double_numbers(iterable):
    double_arr = []
    for i in iterable:
        double_arr.append(i + i)
    return double_arr


# Следующий код сгенерирует и удвоит все значения от 0 до 999999, после чего выведет 0 2 4. Почти миллион значения не пригодился.
for value in double_numbers(range(1000000)):
    print(value)
    if value > 5:
        break


# Мы можем переписать нашу функцию, сделав её генератором с помощью конструкции yield.
def double_numbers_generator(iterable):
    for i in iterable:
        yield i + i


# Теперь программа вычисляет значения из double_numbers_generator(range(1000000)) по одному, 
# прокручивая для каждого цикл, это существенно экономит вычислительные ресурсы.
for value in double_numbers_generator(range(1000000)):  # `test_generator`
    print(value)
    if value > 5:
        break

# Обратите внимание, что range - генератор.

print(range(1000000)) # "range(0, 1000000)"
type(range(1000000)) # <class 'range'>



# Дектораторы

# Декораторы - функции высшего порядка, то есть функции, принимающие и возвращающие функции. 
# Пример - декторатор добавит в возвращаемый функцией список дополнительное значение "Apple".
def add_apple(func):
    def get_food(*args, **kwargs):
        food = func(*args, **kwargs)
        food.append('Apple')
        return food
    return get_food

@add_apple
def only_candy(food):
    eat_only_candy = []
    for product in food:
        if product in ['Candy', 'Cookie', 'Icecream', 'Chocolate']:
            eat_only_candy.append(product)
    return eat_only_candy

# Напечатает: Chocolate, Apple
print(', '.join(only_candy(['Soup', 'Salad', 'Tea', 'Chocolate'])))


# Исключения.
# Некондиционное поведение кода в хороших языках программирования обязано явным для отлаживающего код программиста способом выдавать ошибку. 
# В Python эти ошибки реализуются с помощью механизма исключений (exceptions)

# Например, если мы попробуем поделить на 0
2 / 0 # ZeroDivisionError: division by zero

# Исключения можно обработать в коде более высокого уровня.
# Обработка исключений реализуется с помощью try/except блока

try:
    # Используейте "raise", что бы вызвать исключениеу
    raise IndexError("This is an index error")
except IndexError as e:
    pass  # Ключевое слово pass абсолютно ничего не делает. Всё равно что пустая строка, но с корректным блочным форматированим
except (TypeError, NameError):
    pass  # Можно реагировать сразу на множество типов исключений.
finally:  # Выполнить в любых обстоятельствах
    print("We can clean up resources here")

# Более явно можно показать на примере продвинутой функции деления:

def division(a,b):
    try:
        return a/b
    except ZeroDivisionError:
        return 0


division(4, 2) # 2.0
division(4, 0) # 0

# Оператор assert проверяет переданное в неё условие на истинность и создает исключение AssertionError , если это не так.

x = 2
assert x == 0, 'x не равен 0!' # AssertionError: x не равен 0!

# Это удобно для отлова, например, ситуации, когда в вашу функцию пытаются передать данные за границами её области определения:

def arcsin(x):
    assert -1 <= x <= 1, "Арксинус определен только для [-1,1]!"
    return x*(1 + (1/2)*x**2*(1/3 + 3/4*x**2*(1/5 + 5/6*x**2*(1/7 + 7/8*x**2/9 )))) # первые пять членов разложения арксинуса в ряд Тейлора

arcsin(1/2)*6 # 3.1415111723400297 # дает нам первые 4 цифры pi
arcsin(2) # AssertionError: Арксинус определен только для [-1,1]!



####################################################
# 9. Завершение
####################################################

# В этом кратком пособии мы весьма поверхностно пробежались по основам языка Python. Для более глубокого изучения рекомендую (на русском)
    # https://pythonworld.ru/samouchitel-python
    # Марк Лутц, Программирование на Python
    # Марк Саммерфилд - Python на практике

# Есть две наиболее популярные профессиональные IDE для Python - разработки: PyCharm и VSCode . К сожалению, в связи с политической обстановкой компания JetBrains более не раздает ключи российским студентам. В любом случае, хорошая IDE пригодится, если вы делаете что-то достаточно большое и сложное. 

# Пайтон (официально язык называется именно так) является, наверное, самым популярным в мире скриптовым языком общего назначения. Скриптовые языки - это языки, заточенные на управление некими инструментами более низкого уровня - операционной системой с файлами, библиотеками машинного обучения, анализом данных и т.п.

# Из этого следует, что хорошее знание синтаксиса необходимо но недостаточно для профессионализма, поскольку python здесь является вспомогательным инструметом управления чем-то, нередко, достаточно сложным в своей основе. Перечислим некоторые популярные пути развития, открываемые знанием python:

# Математик / Дата-саентист. Занимается численным моделированием и обработкой данных.
# Обязательные технологии: библиотеки Numpy, Scipy, Jupiter, Pandas, средства рисования графиков.

# Нейросетевик. Машинное обучение и всё, что с ним связано
# Обязательные технологии: всё то же, что и математик + библиотека PyTorch / TensorFlow

# Web-программист. Написание сайтов.
# Обязательные технологии: web-фреймворк (например, flask|django|FastAPI), базы данных SQL, сети, HTML+CSS+JS

# Алгоритмист. Численные алгоритмы низкого уровня, для которых главное - быстродействие.
# Обязательные технологии: другой язык программирования, в котором будет сделана основная реализация алгоритма. Например, C++. Может использовать python в качестве языка прототипирования и проверки идеи. Можно соревноваться на codeforces.com, codewars.com и подобных ресурсах, обычно, без особых проблем, используя python в качестве основного языка. Крайне желательно знание Numpy, Scipy, что бы не терять в качестве.



####################################################
# 10. Zen
####################################################

# Если вы зайдете в REPL Python, и введете import this ,
# вы увидите The Zen of Python - текст, отражающий общую философию языка.
# Вот его перевод на русский:

# Красивое лучше, чем уродливое.
# Явное лучше, чем неявное.
# Простое лучше, чем сложное.
# Сложное лучше, чем запутанное.
# Плоское лучше, чем вложенное.
# Разреженное лучше, чем плотное.
# Читаемость имеет значение.
# Особые случаи не настолько особые, чтобы нарушать правила.
# При этом практичность важнее безупречности.
# Ошибки никогда не должны замалчиваться.
# Если они не замалчиваются явно.
# Встретив двусмысленность, отбрось искушение угадать.
# Должен существовать один и, желательно, только один очевидный способ сделать это.
# Хотя он поначалу может быть и не очевиден, если вы не голландец.
# Сейчас лучше, чем никогда.
# Хотя никогда зачастую лучше, чем прямо сейчас.
# Если реализацию сложно объяснить — идея плоха.
# Если реализацию легко объяснить — идея, возможно, хороша.
# Пространства имён — отличная штука! Будем делать их больше!
