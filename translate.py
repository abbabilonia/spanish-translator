from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model, Model
from keras.layers import Input
import numpy as np
import asyncio

BATCH_SIZE = 64
EPOCHS = 20
LSTM_NODES = 256
NUM_SENTENCES = 20000
MAX_SENTENCE_LENGTH = 50
MAX_NUM_WORDS = 20000
EMBEDDING_SIZE = 100

text_file = 'dataset/spa.txt'

with open(text_file, encoding='utf-8') as f:
    lines = f.read().split("\n")[:-1]

input_sentences = [] # list of inputs
output_sentences = [] # list of outputs
output_sentences_inputs = [] #list of start of sentence
embedding_dictionary = dict() # create an empty dictionary
count = 0

# iterate through the dataset
for line in lines:
    count += 1

    if count > NUM_SENTENCES:
        break

    if '\t' not in line:
        continue

    # separate input and output sentence from the dataset
    input_sentence, output = line.rstrip().split('\t')

    # output sentence (end)
    output_sentence = output + ' <eos>'

    # output sentent (start)
    output_sentence_input = '<sos> ' + output

    # append to created list
    input_sentences.append(input_sentence)
    output_sentences.append(output_sentence)
    output_sentences_inputs.append(output_sentence_input)

# Tokenizing the input words
input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
input_tokenizer.fit_on_texts(input_sentences)
input_integer_seq = input_tokenizer.texts_to_sequences(input_sentences)


# Tokenizing the output words
output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(output_sentences + output_sentences_inputs)
output_integer_seq = output_tokenizer.texts_to_sequences(output_sentences)
output_input_integer_seq = output_tokenizer.texts_to_sequences(output_sentences_inputs)

# count the unique input words in datasets
word2idx_inputs = input_tokenizer.word_index

# count the unique output words in datasets
word2idx_outputs = output_tokenizer.word_index

# count the number of long sentences (output)
max_input_len = max(len(sen) for sen in input_integer_seq)

encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)

# count the number of long sentences (output)
num_words_output = len(word2idx_outputs) + 1
max_out_len = max(len(sen) for sen in output_integer_seq)

idx2word_input = {v:k for k, v in word2idx_inputs.items()}
idx2word_target = {v:k for k, v in word2idx_outputs.items()}

# Translator class
class Translator:
    def __init__(self):
        # self.model = self.get_model()
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.input_sequence = None

    async def get_model(self):
        self.model = load_model('model/lstm_model.h5')
        # self.model.evaluate(verbose=0)
        self.encoder_model = self.encoder()
        self.decoder_model = self.decoder()

    def encoder(self):
        # encoder model
        self.encoder_model_ = Model(self.model.input[0], self.model.get_layer('lstm').output[1:])
        return self.encoder_model_

    def decoder(self):
        # decoder states (ds)
        self.ds_h = Input(shape=(LSTM_NODES,))
        self.ds_c = Input(shape=(LSTM_NODES,))
        self.ds_inputs = [self.ds_h, self.ds_c]

        # decoer Embedding
        # decoder inputs (di)
        self.di_single = Input(shape=(1,))
        self.decoder_embedding = self.model.get_layer('embedding_1')
        self.di_single_x = self.decoder_embedding(self.di_single)

        # decoder LSTM
        self.decoder_lstm = self.model.get_layer('lstm_1')
        self.decoder_outputs, self.h, self.c = self.decoder_lstm(self.di_single_x, initial_state=self.ds_inputs)
        
        # decoder states
        self.decoder_states = [self.h, self.c]

        # decoder dense
        self.decoder_dense = self.model.get_layer('dense')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

        # decoder model
        self.decoder_model_ = Model(
            [self.di_single] + self.ds_inputs,
            [self.decoder_outputs] + self.decoder_states
        )

        return self.decoder_model_

    def preprocess_text(self, input_text):
        # input_text = input_text.lower()
        # input_text = input_text.strip()
        # input_text = input_text.encode('utf8')
        # input_text = input_text.decode('utf8')
        # input_text = input_text.split(' ')

        self.input_seq = input_tokenizer.texts_to_sequences([input_text])[0]
        self.input_seq = pad_sequences([self.input_seq], maxlen=max_input_len)
        return self.input_seq

    async def translate(self, input_text):
        # if self.model == None:
        #     await self.get_model()

        self.input_sequence = self.preprocess_text(input_text)

        self.encoder_states = self.encoder_model.predict(self.input_sequence)
        self.decoder_states_ = self.encoder_states

        self.start_token = np.zeros((1, 1))
        self.start_token[0,0] = word2idx_outputs['<sos>']

        self.output_seq = []

        while True:
            self.decoder_outputs_, self.state_h, self.state_c = self.decoder_model.predict([self.start_token] + self.decoder_states_)
            self.end_token = np.argmax(self.decoder_outputs_[0, -1, :])

            if self.end_token == word2idx_outputs['<eos>']:
                break

            self.word = ''

            if self.end_token > 0:
                self.word = idx2word_target[self.end_token]
                self.output_seq.append(self.word)

            # Update the decoder states
            self.decoder_states_ = [self.state_h, self.state_c]
            
            # Update the start token for the next iteration
            self.start_token[0, 0] = self.end_token

        self.sentence = ' '.join(self.output_seq)

        if "¿" in self.sentence and "." in self.sentence:
            period = self.sentence.index('.')
            self.sentence = self.sentence[:period] + "?" + self.sentence[period+1:]
            return self.sentence

        return self.sentence

# # choice = input('enter choice: ')
# choice = "y"
# while choice == 'y':
#     choice = input('enter choice: ')
#     # input_ex = "Where is my car?"
#     input_ex = input("Enter word: ")
#     translator = Translator()
#     asyncio.run(translator.get_model())

#     translate = asyncio.run(translator.translate(input_ex))
#     print(translate)