### DecisionTree
---

Репозиторий содержит реализацию на Python решающего дерева в виде класса. При инициализации экземпляра класса передаются следующие параметры: максимальная глубина дерева (`max_split`), минимальное количество объектов в узле (`min_samples_split`) и критерий разбиения (`criterion`). Дерево подходит для решения задач регрессии и **бинарной** классификации.

Предусмотрены методы, реализующие следующие критерии разбиения узлов:

- в случае классификации:
    - энтропийный (`entropy`);
    - критерий Джини (`gini`);
- в слуае регрессии:
    - дисперсионный (`variance`) (минимизирующий дисперсию целевой переменной в ветвях);
    - медианный (`median`) (минимизирующий в ветвях модуль абсолютного отклонения целевой переменной от её медианного значения).
    
Для обучения дерева предусмотрен метод `fit`, для получения вектора прогнозов – метод `predict`. Тип задачи (классификация/регрессия) при вызове метода `fit` указывать не требуется: он определяется в соответствии с выбранным критерием разбиения.

В случае классификации выбор наилучшего разбиения осуществляется следующим образом: массив сортируется по выбранному признаку, определяются объекты, соответствующие изменению класса (например, с -1 на 1) и в качестве кандидатов выбираются средние между значениями выбранного признака у таких объектов. При решении задачи регрессии выбор наилучшего разбиения осуществляется путём перебора всех значений всех признаков. 

Вне зависимости от используемого критерия разбиения, наилучшее разбиение выбирается путём подсчёта прироста информации (`information_gain`) и определения разбиения, соответствующего его максимальному значению.

При выполнении метода `fit` на экран выводится процесс его обучения, а именно:

- глубина, на которой находится данный узел;

- признак, по которому происходит разбиение;

- значение, с которым сравнивается значение признака;

- результат сравнения;

- ветка, из которой "пришёл" данный узел (`True` или `False` ветвь);

- объекты, находящиеся в узле.

Для печати процесса обучения требуется установленная библиотека `tabulate`.

С помощью ноутбука `test.ipynb` можно протестировать работу алгоритма.  
