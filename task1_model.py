# For English average 0.32 F1 score, 0.78 Accuracy. Difference caused by the unbalanced dataset.
"""
The approach is the following: for an average text that contains pcl, only some small pieces are actually pcl and the
rest of the text does not. The assumption is that this confuses the model since a combination of pcl and non pcl is
labeled as pcl. To address this, the following approach is used:
 - negative examples are left as they are
 - each positive example is replaced with the actual pieces of pcl inside it that we can get from the categories file
 - the positive examples obtained this way are added with the negative examples to obtain a training dataset
 - the model is trained on it
 - for each text we want to predict, we first use the model on the whole text to get an initial label
 - a window (of the size of the average length of a cleaned pcl fragment * 2) is slided through the text and the model
 is used to predict that particular substring. If it is labeled as pcl, then we consider the whole text as pcl

"""
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
    nlp = None
    # For other languages: python -m spacy download {language_shortcut}_core_news_md
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
        # not starting from 0 because all files have some headers
        # pcl_rows = read_file_rows('./train_subset.csv')[1:]
        # pcl_rows_test = read_file_rows('./validation_subset.csv')[1:]
        # categories_rows = read_file_rows('./dontpatronizeme_categories.tsv', '\t')[5:]

        pcl_rows = read_file_rows(f'./{language_shortcut.upper()}/task1_translated_{language_shortcut}.tsv', '\t')
        random.shuffle(pcl_rows)
        train_len = int(0.8 * len(pcl_rows))
        pcl_rows_test = pcl_rows[train_len:]
        pcl_rows_train = pcl_rows[:train_len]
        categories_rows = read_file_rows(f'./{language_shortcut.upper()}/task2_translated_{language_shortcut}.tsv',
                                         '\t')

        positive_examples = []
        negative_examples = []
        test_data = []
        test_labels = []

        for curr_row in pcl_rows_train:
            if int(curr_row[-2]) == 0:
                negative_examples.append(clean_sentence(curr_row[-3]))
            else:
                for category_row in categories_rows:
                    if curr_row[0] == category_row[0]:
                        positive_examples.append(clean_sentence(category_row[-3]))

        for curr_row in pcl_rows_test:
            test_data.append(clean_sentence(curr_row[-3]))
            test_labels.append(int(curr_row[-2]))

        return positive_examples, negative_examples, test_data, test_labels


    (positive_examples, negative_examples, test_data, test_labels) = build_examples()

    negative_examples = negative_examples[0: len(positive_examples)]

    # Getting the average length of a positive example
    avg_pos_len = 0

    for positive_example in positive_examples:
        avg_pos_len += len(positive_example)

    avg_pos_len = int(avg_pos_len / len(positive_examples))

    # Adding together the positive and negative examples to get training data
    train_examples = []
    train_labels = []
    for negative_example in negative_examples:
        train_examples.append(negative_example)
        train_labels.append(0)

    for positive_example in positive_examples:
        train_examples.append(positive_example)
        train_labels.append(1)


    def get_model():
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

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
            tf.keras.layers.Dense(2)
        ])
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


    train_examples = np.array(train_examples)
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
    window = avg_pos_len * 2
    for q in range(0, len(test_data)):
        print(f"Example {q}")
        test_example = test_data[q]
        if len(test_example) < 10:
            predicted_labels.append(0)
            continue
        predicted_label = model.predict([test_example])[0].argmax()
        # Sliding a window through the text to look for pcl
        for i in range(0, len(test_example) - window, window):
            test_example_shard = test_example[i:(i + window)]
            shard_label = model.predict([test_example_shard])[0].argmax()
            if shard_label == 1:
                # if a piece of the text is considered pcl, then the whole text should be labeled as pcl
                predicted_label = 1
                break
        predicted_labels.append(int(predicted_label))
    curr_f1_score = f1_score(test_labels, predicted_labels)
    curr_accuracy_score = accuracy_score(test_labels, predicted_labels)
    print(f'F1 score for validation is {curr_f1_score}')
    print(f'Accuracy score for validation is {curr_accuracy_score}')
    results.append((language, curr_f1_score, curr_accuracy_score))

with open('task1_results.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['english:'])
    writer.writerow(['F1 score for validation is 0.32'])
    writer.writerow(['Accuracy score for validation is 0.78'])
    for result in results:
        writer.writerow([f'{result[0]}:'])
        writer.writerow([f'F1 score for validation is {result[1]}'])
        writer.writerow([f'Accuracy score for validation is {result[2]}'])
