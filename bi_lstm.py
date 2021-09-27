# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Train keras model to detect catgory in bbc text.
import csv
import numpy
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameters.
token_size = 3000
embedding_dim = 16
text_length = 120
epochs = 15
training_size = 2000
batchSize = 64
# Stop words.
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

# Save as sentences and lables.
sentences = []
labels = []

# Get data from bbc-text.csv file.
with open("./bbc-text.csv", 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
        sentences.append(sentence)

# Prepare training data.
training_sentences = sentences[0:training_size]
training_labels_words = labels[0:training_size]
# Prepare validation data.
validation_sentences = sentences[training_size:]
validation_labels_words = labels[training_size:]

# Create tokenizer.
tokenizer = Tokenizer(num_words=token_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Get text from word tokens.
reverse_word_index = dict([(value, key)
                          for (key, value) in word_index.items()])


def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '') for i in text])


# Create training sequences.
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_data = numpy.array(pad_sequences(
    training_sequences, maxlen=text_length, padding="post", truncating="post"))

# Create validation sequences.
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_data = numpy.array(pad_sequences(
    validation_sequences, maxlen=text_length, padding="post", truncating="post"))

# Create label tokens.
labels_tokenizer = Tokenizer()
labels_tokenizer.fit_on_texts(labels)
labels_word_index = labels_tokenizer.word_index
training_labels = numpy.array(
    labels_tokenizer.texts_to_sequences(training_labels_words))
validation_labels = numpy.array(
    labels_tokenizer.texts_to_sequences(validation_labels_words))

# Get label text from word tokens.
reverse_labels_word_index = dict([(value, key)
                          for (key , value) in labels_word_index.items()])

# Create model with 1 output unit for classification.
model = keras.Sequential([
    keras.layers.Embedding(token_size, embedding_dim,
                           input_length=text_length),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(6, activation="softmax")
])

# Set loss function and optimizer.
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      # Stop when validation accuracy is more than 98%.
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.98:
            print("\nTraining Stopped!")
            self.model.stop_training = True


# Callback function to check accuracy.
checkAccuracy = myCallback()

# Train model.
model.fit(training_data, training_labels, epochs=epochs, callbacks=[
          checkAccuracy], batch_size=batchSize, validation_data=(validation_data, validation_labels), verbose=1)


# Predict on a random validation text.
index = 7
text = validation_data[index]
prediction = model.predict(text.reshape(1, text_length, 1))

# Get max value.
label = 0
for i in range(6):
  if prediction[0][label] < prediction[0][i]:
    label = i

# Show prediction.
print("Prediciton : ", reverse_labels_word_index.get(label))
print("Label : ", validation_labels_words[index])
print("Text : ", decode_sentence(validation_data[index]))
