import csv
# necessary for tensorflow_hub, even if not directly used
import tensorflow_text
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
import random
import stopwords
import spacy

results = []
for language, language_shortcut in [('dutch', 'nl'), ('french', 'fr'), ('italian', 'it'), ('german', 'de'), ('spanish', 'es')]:
    nlp = spacy.load(f'{language_shortcut}_core_news_md')

    stop_words = stopwords.get_stopwords(f'{language}')

    # removing punctuation and lemmatizing
    def clean_sentence(sentence):
        sentence = sentence.lower()
        for char in sentence:
            if (not char.isalpha()) and char != ' ':
                sentence = sentence.replace(char, ' ')

        tokens = sentence.split(' ')
        good_tokens = []
        for token in tokens:
            if token not in stop_words:
                good_tokens.append(token)
        sentence = ' '.join(good_tokens)
        tokens = nlp(sentence)

        sentence = ' '.join([x.lemma_ for x in tokens])
        return sentence


    # reading rows from files
    def read_file_rows(file_path, delimiter=","):
        rows = []
        tsv_file = open(file_path)
        tsv_reader = csv.reader(tsv_file, delimiter=delimiter)

        for row in tsv_reader:
            if len(row) > 2:
                rows.append(row)
        return rows


    # building lists of positive examples, negative examples, test data and test labels
    def build_examples():
        categories_rows = read_file_rows(f'./{language_shortcut.upper()}/task2_translated_{language_shortcut}.tsv',
                                         '\t')
        random.shuffle(categories_rows)
        train_len = int(0.8 * len(categories_rows))
        categories_rows_test = categories_rows[train_len:]
        categories_rows_train = categories_rows[:train_len]

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        for curr_row in categories_rows_train:
            train_data.append(clean_sentence(curr_row[-3]))
            train_labels.append(int(curr_row[-2]))

        for curr_row in categories_rows_test:
            test_data.append(clean_sentence(curr_row[-3]))
            test_labels.append(int(curr_row[-2]))

        return train_data, train_labels, test_data, test_labels


    (train_data, train_labels, test_data, test_labels) = build_examples()
    classes_list = list(set(test_labels))
    classes_number = len(list(set(test_labels)))
    train_labels = [classes_list.index(train_label) for train_label in train_labels]
    test_labels = [classes_list.index(test_label) for test_label in test_labels]


    def get_model():
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))

        BUFFER_SIZE = 10000
        BATCH_SIZE = 64

        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        VOCAB_SIZE = 1000
        encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_mode="int")
        encoder.adapt(train_dataset.map(lambda text, label: text))
        hub_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3", input_shape=[],
                                   output_shape=[512, 16],
                                   dtype=tf.string, trainable=True)
        model = tf.keras.models.Sequential([
            hub_layer,
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(classes_number)
        ])
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


    train_examples = np.array(train_data)
    train_labels = np.array(train_labels).astype('float32')

    model = get_model()
    history = model.fit(train_examples, train_labels, batch_size=64, epochs=10)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()

    predicted_labels = []
    for q in range(0, len(test_data)):
        print(f"Example {q}")
        test_example = test_data[q]
        if len(test_example) < 10:
            predicted_labels.append('Compassion')
            continue
        predicted_label = model.predict([test_example])[0].argmax()
        predicted_labels.append(predicted_label)
    curr_f1_score = f1_score(test_labels, predicted_labels)
    curr_accuracy_score = accuracy_score(test_labels, predicted_labels)
    print(f'F1 score for validation is {curr_f1_score}')
    print(f'Accuracy score for validation is {curr_accuracy_score}')
    results.append((language, curr_f1_score, curr_accuracy_score))

with open('task2_results.csv') as file:
    writer = csv.writer(file, delimiter=',')
    for result in results:
        writer.writerow(f'{result[0]}:')
        writer.writerow(f'F1 score for validation is {result[1]}')
        writer.writerow(f'Accuracy score for validation is {result[2]}')
