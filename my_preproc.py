import sys #нужен для работы с sys.modules (регистрация заглушки pkg_resources)
import string #содержит string.punctuation (набор ASCII-знаков препинания)
import re #регулярные выражения (удаление цифр, проверка русских слов)

#Совместимость с Python 3.12+ для Natasha, иначе не получилось запустить
try:                                          # Пытаемся импортировать pkg_resources — он нужен Natasha
    import pkg_resources                      # Если модуль есть — используем его напрямую
except ModuleNotFoundError:                   # Если pkg_resources отсутствует (например, в Python 3.12+),
    import types                              # импортируем types — пригодится для создания фиктивного модуля
    import importlib.metadata                 # импортируем importlib.metadata — современный API для entry points

    pkg_resources = types.ModuleType("pkg_resources")  # Создаём фиктивный модуль с именем "pkg_resources"

    def iter_entry_points(group, name=None):  # Функция, имитирующая pkg_resources.iter_entry_points
        return [ep for ep in importlib.metadata.entry_points(group=group)]  # Возвращаем entry points из importlib.metadata

    class DummyWorkingSet:                    # Класс-заглушка, имитирующий pkg_resources.WorkingSet
        def __init__(self, *args, **kwargs):  # Конструктор принимает любые аргументы и ничего не делает
            pass

        def iter_entry_points(self, group, name=None):  # Метод, имитирующий одноимённый метод WorkingSet
            return iter_entry_points(group, name)       # Делегирует вызов глобальной функции iter_entry_points

    pkg_resources.iter_entry_points = iter_entry_points  # Привязываем функцию к фиктивному модулю
    pkg_resources.WorkingSet = DummyWorkingSet           # Привязываем класс-заглушку к фиктивному модулю
    sys.modules["pkg_resources"] = pkg_resources         # Регистрируем заглушку в sys.modules, чтобы `import pkg_resources` работал
# -------------------------------------------------

import nltk #библиотека для обработки текста
import pymorphy3 #морфологический анализатор русского языка
import spacy #NLP-библиотека

from nltk.corpus import stopwords #Импорт стоп-слов из корпуса NLTK
from nltk.stem.snowball import SnowballStemmer #Импорт Snowball-стеммера из NLTK (русский язык)
from nltk.tokenize.regexp import WordPunctTokenizer #Импорт токенизатора, отделяющего пунктуацию от слов
from nltk.tokenize import word_tokenize #Импорт стандартного токенизатора NLTK

from natasha import ( #Импорт компонентов Natasha:
    Segmenter, #Segmenter — сегментация текста на токены
    MorphVocab, #MorphVocab — морфологический словарь
    NewsEmbedding, #NewsEmbedding — эмбеддинги, обученные на новостях
    NewsMorphTagger, #NewsMorphTagger — морфологический теггер
    Doc #Doc — объект документа Natasha
)

# Ресурсы NLTK
nltk.download('stopwords', quiet=True) #корпус стоп-слов
nltk.download('punkt', quiet=True) #модель токенизации punkt
nltk.download('punkt_tab', quiet=True) #таблицы для punkt (нужно новым версиям NLTK)

# Инициализация инструментов
tokenizer = WordPunctTokenizer() #токенизатор, отделяющий пунктуацию от слов
morph = pymorphy3.MorphAnalyzer() #морфологический анализатор PyMorphy3
nlp = spacy.load("ru_core_news_md") #Загружаем среднюю русскую модель spaCy
stemmer = SnowballStemmer("russian")  #Создаём Snowball-стеммер для русского языка

# Natasha
segmenter = Segmenter() #Создаём сегментатор Natasha
morph_vocab = MorphVocab()  #Создаём морфологический словарь Natasha
emb = NewsEmbedding() #Загружаем новостные эмбеддинги
morph_tagger = NewsMorphTagger(emb)  #Создаём морфологический теггер на основе эмбеддингов

sentences = [
    "Я искренне рад, что эта потрясающая команда выиграла сложнейший матч со счётом 3:0!",  #позитив
    "Ужасный сервис, еда была абсолютно отвратительной и холодной.",                        #негатив
    "В среду в главном корпусе университета состоится очередное собрание в 14:30."          #нейтраль
]

RU_WORD_MATCH = re.compile(r'^[а-яё]+$') #Регулярное выражение: вся строка целиком состоит из строчных русских букв


# Функции предобработки
def lowercase_text(text): #приведение текста к нижнему регистру
    return text.lower()

def remove_num(text):#удаление цифр
    return re.sub(r'\d+', '', text) #Заменяет все последовательности цифр на пустую строку

def rem_punct(text): #удаление ASCII-пунктуации
    translator = str.maketrans('', '', string.punctuation) #Создаёт таблицу перевода, удалить все символы string.punctuation
    return text.translate(translator)

def remove_stopwords_and_punct(text):  #удаление стоп-слов, не-русских и коротких токенов
    stop_words = set(stopwords.words("russian")) - {'не', 'нет'} #Набор стоп-слов без «не» и «нет»
    word_tokens = tokenizer.tokenize(text) #Токенизируем текст

    filtered_tokens = [] #Пустой список для отфильтрованных токенов
    for word in word_tokens:  #Проходим по каждому токену
        if word in stop_words: #Если токен — стоп-слово,
            continue  #пропускаем его
        if not RU_WORD_MATCH.match(word): #Если токен не чисто русское строчное слово,
            continue  #пропускаем его
        if len(word) <= 2 and word not in ['не', 'рад']: #Если токен короче 3 символов и не «не»/«рад»,
            continue #пропускаем его
        filtered_tokens.append(word) #Иначе добавляем токен в результат

    return filtered_tokens #Возвращаем список отфильтрованных токенов

def s_words(text): #Функция стемминга через Snowball (NLTK)
    word_tokens = word_tokenize(text)  #Токенизируем текст
    stems = [stemmer.stem(word) for word in word_tokens if word.isalpha()] #Стеммим только буквенные токены
    return stems

def lemmatize_spacy(text): #Функция лемматизации через spaCy
    doc = nlp(text)  #Прогоняем текст через модель spaCy
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha] # Леммы не-стоп-слов и буквенных токенов
    return lemmas

def full_preprocess_pipeline(text, method='natasha'): #Главный пайплайн предобработки с выбором метода
    lowered = lowercase_text(text) #Приводим текст к нижнему регистру
    no_num = remove_num(lowered) #Удаляем все цифры
    clean_text = rem_punct(no_num) #Удаляем ASCII-пунктуацию

    if method == 'spacy':
        return lemmatize_spacy(clean_text) #возвращаем леммы spaCy

    processed_tokens = remove_stopwords_and_punct(clean_text)  #Общая фильтрация: стоп-слова, не-русские, короткие
    result = [] #Пустой список для итоговых токенов

    if method == 'natasha':
        clean_str = " ".join(processed_tokens) #склеиваем отфильтрованные токены обратно в строку
        doc = Doc(clean_str) #Создаём объект Doc Natasha
        doc.segment(segmenter) #Сегментируем текст на токены
        doc.tag_morph(morph_tagger) #Размечаем токены морфологически

        for t in doc.tokens: #Проходим по всем токенам
            t.lemmatize(morph_vocab) #Лемматизируем токен через словарь
            result.append(t.lemma) #Добавляем лемму в результат

    elif method == 'pymorphy3':
        for t in processed_tokens: #проходим по отфильтрованным токенам
            result.append(morph.parse(t)[0].normal_form) #Берём первый разбор и его нормальную форму

    elif method == 'stemming':
        for t in processed_tokens: #проходим по отфильтрованным токенам
            result.append(stemmer.stem(t)) #Стеммим токен и добавляем в результат

    return result #Возвращаем итоговый список

#Вывод результата
print("Сравнение результатов предобработки")
for i, s in enumerate(sentences, 1):
    print(f"\nПредложение {i}: '{s}'")
    print(f"1. Natasha:          {full_preprocess_pipeline(s, method='natasha')}")
    print(f"2. PyMorphy3:        {full_preprocess_pipeline(s, method='pymorphy3')}")
    print(f"3. SpaCy:            {full_preprocess_pipeline(s, method='spacy')}")
    print(f"4. NLTK stemming:    {full_preprocess_pipeline(s, method='stemming')}")