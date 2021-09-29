from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Заполните генератор случайных чисел, чтобы он генерировал те же числа
        # каждый раз при запуске программы.
        random.seed(1)

        # Мы моделируем одиночный нейрон с 3 входными подключениями и 1 выходным подключением.
        # Мы назначаем случайные веса матрице 3 x 1 со значениями в диапазоне от -1 до 1
        # и означает 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1


    # Сигмоидальная функция, которая описывает S-образную кривую.
    # Мы передаем взвешенную сумму входных данных через эту функцию в
    # нормализовать их между 0 и 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Производная сигмовидной функции.
    # Это градиент сигмовидной кривой.
    # Это показывает, насколько мы уверены в существующем весе.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)


    # Мы обучаем нейронную сеть методом проб и ошибок.
    # Каждый раз корректировать синаптические веса.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Пропустить обучающий набор через нашу нейронную сеть (отдельный нейрон).
            output = self.think(training_set_inputs)

            # Вычислить ошибку (Разница между желаемым результатом
            # и прогнозируемый результат).
            error = training_set_outputs - output

            # Умножьте ошибку на вход и снова на градиент сигмовидной кривой.
            # Это означает, что менее достоверные веса корректируются больше.
            # Это означает, что входы, которые равны нулю, не вызывают изменения весов.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))


            # Отрегулируйте веса.
            self.synaptic_weights += adjustment

    # Нейронная сеть думает.
    def think(self, inputs):
        # Передаем входные данные через нашу нейронную сеть (наш единственный нейрон).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    # Изолировать нейронную сеть с одним нейроном.
    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)

    # Обучающий набор. У нас есть 4 примера, каждый из 3 входных значений
    # и 1 выходное значение.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Обучить нейронную сеть с помощью обучающего набора.
    # Сделайте это 10 000 раз и каждый раз вносите небольшие поправки.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)

    # Протестируйте нейронную сеть в новой ситуации.
    print ("Considering new situation [1, 0, 0] -> ?: ")
    print (neural_network.think(array([1, 0, 0])))
