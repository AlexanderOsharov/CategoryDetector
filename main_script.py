import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from flask import Flask, request, jsonify

app = Flask(__name__)
pipeline = None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']
    predictions = predict_categories(pipeline, text)
    return jsonify(predictions)

# Используем классификатор для определения категорий нового текста
def predict_categories(pipeline, new_text, top_n=3):
    if not pipeline:  # проверка на то, что модель обучена
        return []

    probabilities = pipeline.predict_proba([new_text])[0]
    top_categories = sorted(zip(pipeline.classes_, probabilities), key=lambda x: -x[1])[:top_n]
    return [category for category, prob in top_categories]


def main():
    # Загружаем датасет с примерами текстов и их категориями (необходимо составить датасет)
    data = [
        # Тексты про мемы
        ("Смешная картинка с текстом про кошек", "memes"),
        ("когда ты выпил кофе, но все равно сонный", "memes"),
        ("Funny picture with text about dogs", "memes"),
        ("When you finish work, but still have more work", "memes"),

        # Тексты про новости
        ("Новый закон о блокировке интернет-ресурсов вступил в силу", "news"),
        ("Президент ввел санкции против некоторых компаний", "news"),
        ("Прогноз погоды на завтра: тепло и солнечно", "news"),
        ("Ожидается обильный снегопад в понедельник", "news"),
        ("New law on internet resource blocking comes into force", "news"),
        ("President imposes sanctions on some companies", "news"),
        ("Weather forecast for tomorrow: warm and sunny", "news"),
        ("Heavy snowfall expected on Monday", "news"),

        # Тексты про игры
        ("Компания выпускает обновление для популярной игры", "games"),
        ("Поклонники с нетерпением ждут новую часть игры", "games"),
        ("Company releases update for popular game", "games"),
        ("Fans eagerly await the new installment of the game", "games"),

        # Тексты про фильмы
        ("Актер поделился фото с съемок нового фильма", "films"),
        ("Продюсер анонсировал продолжение культового фильма", "films"),
        ("Actor shares photo from the set of a new film", "films"),
        ("Producer announces sequel to cult film", "films"),

        # Тексты про еду
        ("Рецепт вкусного домашнего пирога с яблоками", "meal"),
        ("Топ-5 блюд, которые легко приготовить после работы", "meal"),
        ("Recipe for a delicious homemade apple pie", "meal"),
        ("Top 5 dishes that are easy to prepare after work", "meal"),

        # Тексты про книги
        ("Опубликован список лучших книг 2021 года", "books"),
        ("Автор рассказал об идее новой книги", "books"),
        ("The list of the best books of 2021 has been published", "books"),
        ("The author talks about the for a new book", "books"),

        # Тексты про животных
        ("Как ухаживать за домашним питомцем: советы ветеринара", "animals"),
        ("Любопытные факты о поведении диких животных", "animals"),
        ("How to care for a pet: veterinarian tips", "animals"),
        ("Interesting facts about the behavior of wild animals", "animals"),

        # Тексты про психологию
        ("5 способов справиться со стрессом на работе", "psychology"),
        ("Зачем нам нужно понимать свои эмоции", "psychology"),
        ("5 ways to deal with stress at work", "psychology"),
        ("Why we need to understand our emotions", "psychology"),

        # Тексты про науку
        ("Ученые обнаружили новый вид динозавра", "sciences"),
        ("Исследование показывает связь между питанием и здоровьем", "sciences"),
        ("Scientists discover a new species of dinosaur", "sciences"),
        ("Study shows link between diet and health", "sciences"),

        # Тексты про мультфильмы
        ("Студия анонсировала новый мультфильм для всей семьи", "cartoons"),
        ("Топ-10 незабываемых мультфильмов детства", "cartoons"),
        ("Studio announces new family-friendly animated film", "cartoons"),

        # Тексты про парфюмерию (perfumery)
        ("Лучшие ароматы для осени: обзор новинок", "perfumery"),
        ("Как выбрать подходящий парфюм для работы", "perfumery"),
        ("Best fall fragrances: a review of new products", "perfumery"),
        ("How to choose the right perfume for work", "perfumery"),

        # Тексты про одежду (clothes)
        ("Тренды моды: какие платья будут популярны в этом сезоне", "clothes"),
        ("Советы по уходу за хлопковыми изделиями", "clothes"),
        ("Fashion trends: what dresses will be popular this season", "clothes"),
        ("Tips for caring for cotton garments", "clothes"),

        # Тексты про товары для дома (household items)
        ("Как выбрать подходящее постельное белье для дома", "household items"),
        ("Очистители воздуха: устройства и их виды", "household items"),
        ("How to choose the right bedding for your home", "household items"),
        ("Air purifiers: devices and types", "household items"),

        # Тексты про канцелярию (chancellery)
        ("Как выбрать удобные кресло и стол для работы", "chancellery"),
        ("Топ-5 популярных брендов ручек", "chancellery"),
        ("How to choose a comfortable chair and desk for work", "chancellery"),
        ("Top 5 popular pen brands", "chancellery"),

        # Тексты про садоводство (gardening)
        ("Советы по уходу за дачным участком после зимы", "gardening"),
        ("Как правильно вырастить помидоры на своем участке", "gardening"),
        ("Tips for caring for your garden plot after winter", "gardening"),
        ("How to grow tomatoes correctly in your garden", "gardening"),
    ]

    texts, categories = zip(*data)

    # Разделяем датасет на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(texts, categories, test_size=0.2)

    # Создаем и обучаем классификатор
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(stop_words='english')),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(X_train, y_train)

    # Оценить классификатор на тестовой выборке
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    return pipeline

if __name__ == '__main__':
    # Train my model
    pipeline = main()

    # Run the Flask server
    app.run(debug=True)