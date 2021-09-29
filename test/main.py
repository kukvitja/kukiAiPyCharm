import torch
import torch.nn as nn
import spacy


from func import *

import pandas as pd

from collections import Counter
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, review_df, vectorizer):
        """
        Аргументы:
        review_df (pandas.DataFrame): набор данных vectorizer
        (ReviewVectorizer): экземпляр векторизатора, полученный
        на основе набора данных
        """
        self.review_df = review_df
        self._vectorizer = vectorizer
        self.train_df = self.review_df[self.review_df.split=='train']
        self.train_size = len(self.train_df)
        self.val_df = self.review_df[self.review_df.split=='val']
        self.validation_size = len(self.val_df)
        self.test_df = self.review_df[self.review_df.split == 'test']
        self.test_size = len(self.test_df)
        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}
        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
        """Загружает набор данных и создает новый векторизатор с нуля
        Аргументы:
        review_csv (str): местоположение набора данных
        Возвращает:
        экземпляр ReviewDataset
        """

        review_df = pd.read_csv(review_csv)
        return cls(review_df, ReviewVectorizer.from_dataframe(review_df))

    def get_vectorizer(self):
        """ возвращает векторизатор """

        return self._vectorizer

    def set_split(self, split="train"):
        """ выбор фрагментов набора данных по столбцу из объекта dataframe
        Аргументы:
        split (str): "train" (обучающий), "val" (проверочный)
        или "test" (контрольный)
        """

        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """ основной метод-точка входа для наборов данных PyTorch
        Аргументы:
        index (int): индекс точки данных
        Возвращает:
        словарь признаков (x_data) и метки (y_target) точки данных
        """

        row = self._target_df.iloc[index]
        review_vector = \
            self._vectorizer.vectorize(row.review)
        rating_index = \
            self._vectorizer.rating_vocab.lookup_token(row.rating)
        return {'x_data': review_vector,
                'y_target': rating_index}

    def get_num_batches(self, batch_size):
        """ Возвращает по заданному размеру пакета число пакетов в наборе
        Аргументы:
        batch_size (int)
        Возвращает:
        число пакетов в наборе данных
        """

        return len(self) // batch_size




"""
    Класс Vocabulary,
    предназначенный для хранения соответствия токенов целым числам,
    необходимого остальной части конвейера машинного обучения
"""
class Vocabulary(object):
    """ Класс, предназначенный для обработки текста и извлечения Vocabulary
    для отображения
    """
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        Аргументы:
        token_to_idx (dict): готовый ассоциативный массив соответствий
        токенов индексам add_unk (bool): флаг, указывающий,
        нужно ли добавлять токен UNK
        unk_token (str): добавляемый в словарь токен UNK
        """
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token
                                for token, idx in self._token_to_idx.items()}
        self._add_unk = add_unk
        self._unk_token = unk_token
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)
    def to_serializable(self):
        """ Возвращает словарь с возможностью сериализации """
        return {'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk,
                'unk_token': self._unk_token}
    @classmethod
    def from_serializable(cls, contents):
        """ Создает экземпляр Vocabulary на основе сериализованного словаря """

        return cls(**contents)

    def add_token(self, token):
        """ Обновляет словари отображения, добавляя в них токен.
        Аргументы:
        token (str): добавляемый в Vocabulary элемент
        Возвращает:
        index (int): соответствующее токену целочисленное значение
        """

        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        """ Извлекает соответствующий токену индекс
        или индекс UNK, если токен не найден.
        Аргументы:
        token (str): токен для поиска
        Возвращает:
        index (int): соответствующий токену индекс
        Примечания:
        'unk_index' должен быть >=0 (добавлено в Vocabulary)
        для должного функционирования UNK
        """

        if self.add_unk:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """ Возвращает соответствующий индексу токен
        Аргументы:
        index (int): индекс для поиска
        Возвращает:
        token (str): соответствующий индексу токен
        Генерирует:
        KeyError: если индекс не найден в Vocabulary
        """

        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


"""
    Класс Vectorizer, преобразующий текст в числовые векторы
"""
class ReviewVectorizer(object):
    """ Векторизатор, приводящий словари в соответствие друг другу
        и использующий их
    """
    def __init__(self, review_vocab, rating_vocab):
        """
        Аргументы:
            review_vocab (Vocabulary): отображает слова
            в целочисленные значения
            rating_vocab (Vocabulary): отображает метки классов
            в целочисленные значения
        """
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review):
        """ Создает свернутый унитарный вектор для обзора
        Аргументы:
            review (str): обзор
            Возвращает:
            one_hot (np.ndarray): свернутое унитарное представление
        """

        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)
        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1
        return one_hot

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        """ Создает экземпляр векторизатора на основе
            объекта DataFrame набора данных
        Аргументы:
            review_df (pandas.DataFrame): набор данных обзоров
            cutoff (int): параметр для фильтрации по частоте вхождения
        Возвращает:
            экземпляр класса ReviewVectorizer
        """

        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)
        # Добавить рейтинги
        print(set(review_df))
        for rating in sorted(set(review_df.Account)):
            rating_vocab.add_token(rating)
        # Добавить часто встречающиеся слова, если число вхождений
        # больше указанного
        word_counts = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1
        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)
        return cls(review_vocab, rating_vocab)

    @classmethod
    def from_serializable(cls, contents):
        """ Создает экземпляр ReviewVectorizer на основе
            сериализуемого словаря
        Аргументы:
            contents (dict): сериализуемый словарь
        Возвращает:
            экземпляр класса ReviewVectorizer
        """
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])
        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

    def to_serializable(self):
        """ Создает сериализуемый словарь для кэширования
        Возвращает:
        contents (dict): сериализуемый словарь
        """

        return {'review_vocab': self.review_vocab.to_serializable(),
                'rating_vocab': self.rating_vocab.to_serializable()}


"""
    Генерация мини-пакетов на основе набора данных
"""
def generate_batches(dataset, batch_size, shuffle=True,
    drop_last=True, device="cpu"):
    """
        Функция-генератор — адаптер для объекта DataLoader фреймворка PyTorch.
        Гарантирует размещение всех тензоров на нужном устройстве.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict




"""
    __________________________________________________________________________________
"""

"""
    Классификатор на основе перцептрона
"""
import torch.nn as nn
import torch.nn.functional as F


class ReviewClassifier(nn.Module):
    """ Простой классификатор на основе перцептрона """
    def __init__(self, num_features):
        """
            Аргументы:
                num_features (int): размер входного вектора признаков
        """
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features,
                            out_features=1)
    def forward(self, x_in, apply_sigmoid=False):
        """ Прямой проход классификатора
            Аргументы:
                x_in (torch.Tensor): входной тензор данных
                x_in.shape должен быть (batch, num_features)
                apply_sigmoid (bool): флаг для сигма-функции активации
                при использовании функции потерь на основе перекрестной
                энтропии должен равняться false
            Возращает:
                итоговый тензор. tensor.shape должен быть (batch,).
        """
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = F.sigmoid(y_out)
        return y_out


"""
    ------------------------------------------------------------
"""

"""
    Гиперпараметры и настройки программы для классификатора обзоров Yelp на основе перцептрона
"""
from argparse import Namespace

args = Namespace(
        # Информация о данных и путях
        frequency_cutoff=25,
        model_state_file='model.pth',
        review_csv='data/data.csv',
        save_dir='model_storage/ch3/yelp/',
        vectorizer_file='vectorizer.json',
        # Гиперпараметры модели отсутствуют
        # Гиперпараметры обучения
        batch_size=128,
        early_stopping_criteria=5,
        learning_rate=0.001,
        num_epochs=100,
        seed=1337,
        # Настройки времени выполнения не приводятся для экономии места
    )


"""
    -------------------------------------------------
"""
"""Создание набора данных, модели, функции потерь, оптимизатора и состояния обучения"""
import torch.optim as optim

def make_train_state(args):
    return {'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1}


train_state = make_train_state(args)

if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")

# Набор данных и векторизатор
dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
vectorizer = dataset.get_vectorizer()

# Модель
classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
classifier = classifier.to(args.device)

# Функция потерь и оптимизатор
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)



"""
    ----------------------------------------------------------------------------------------- 
"""
"""
    Простейший цикл обучения
"""
for epoch_index in range(args.num_epochs):
    train_state['epoch_index'] = epoch_index

    # Проход в цикле по обучающему набору данных

    # Настройки: создаем генератор пакетов, устанавливаем значения
    # переменных loss и acc равными 0, включаем режим обучения
    dataset.set_split('train')
    batch_generator = generate_batches(dataset,
                                        batch_size=args.batch_size,
                                        device=args.device)
    running_loss = 0.0
    running_acc = 0.0
    classifier.train()
    for batch_index, batch_dict in enumerate(batch_generator):
        # Процедура обучения состоит из пяти шагов:

        # Шаг 1. Обнуляем градиенты
        optimizer.zero_grad()

        # Шаг 2. Вычисляем выходные значения
        y_pred = classifier(x_in=batch_dict['x_data'].float())

        # Шаг 3. Вычисляем функцию потерь
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_index + 1)

        # Шаг 4. Получаем градиенты на основе функции потерь
        loss.backward()

        # Шаг 5. Оптимизатор обновляет значения параметров по градиентам
        optimizer.step()

        # -----------------------------------------
        # Вычисляем точность
        acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_batch - running_acc) / (batch_index + 1)

    train_state['train_loss'].append(running_loss)
    train_state['train_acc'].append(running_acc)

    # Проход в цикле по проверочному набору данных

    # Настройки: создаем генератор пакетов, устанавливаем значения
    # переменных loss и acc равными 0, включаем режим проверки
    dataset.set_split('val')
    batch_generator = generate_batches(dataset,
                                       batch_size=args.batch_size,
                                       device=args.device)
    running_loss = 0.
    running_acc = 0.
    classifier.eval()
    for batch_index, batch_dict in enumerate(batch_generator):

        # Шаг 1. Вычисляем выходные значения
        y_pred = classifier(x_in=batch_dict['x_data'].float())

        # Шаг 2. Вычисляем функцию потерь
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_index + 1)

        # Шаг 3. Вычисляем точность
        acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_batch - running_acc) / (batch_index + 1)
    train_state['val_loss'].append(running_loss)
    train_state['val_acc'].append(running_acc)