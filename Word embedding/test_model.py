from gensim.models import Word2Vec

# 1. Загружаем (открываем) ранее сохраненную модель
# Файлы homograph_w2v.model.trainables.syn1neg.npy и .wv.vectors.npy
# должны лежать в этой же папке рядом!
model = Word2Vec.load("homograph_w2v.model")

# 2. Посмотрим, какие вообще слова знает модель (размер словаря)
vocab_size = len(model.wv)
print(f"Количество слов в словаре модели: {vocab_size}")

# 3. Извлечем вектор конкретного слова (массив из 50 чисел, который мы обучили)
word_vector = model.wv['замо`к']
print("\nПример вектора для слова 'замо`к' (первые 5 координат):")
print(word_vector[:5])

# 4. Проверим косинусное сходство между омографом и контекстным словом
similarity_score = model.wv.similarity('замо`к', 'двери')
print(f"\nБлизость слова 'замо`к' к слову 'двери': {similarity_score:.4f}")
