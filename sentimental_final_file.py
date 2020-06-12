from keras.models import load_model
model = load_model('output.h5')
model.summary()

from flask import Flask, request
app = Flask(__name__)

#%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
# from tf.keras.models import Sequential  # This does not work!
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import download
import imdb
imdb.maybe_download_and_extract()

x_train_text, y_train = imdb.load_data(train=True) #loading train data
x_test_text, y_test = imdb.load_data(train=False) # loading test data
print("Train-set size: ", len(x_train_text))
print("Test-set size:  ", len(x_test_text))

data_text = x_train_text + x_test_text
num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data_text)
tokenizer.word_index

x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
x_test_tokens = tokenizer.texts_to_sequences(x_test_text) # converting text data into tokens
#print(x_train_text[2])
#print(np.array(x_train_tokens[0]))
num_tokens = np.array([len(tokens) for tokens in x_train_tokens + x_test_tokens])
print(np.mean(num_tokens))
print(np.max(num_tokens))
print(np.min(num_tokens))

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
print(max_tokens)
print(str(np.sum(num_tokens < max_tokens) / len(num_tokens) * 100) +' %' )

pad = 'pre'
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)
num_tokens_pad = np.array([len(tokens) for tokens in x_train_pad + x_test_pad])
result = model.evaluate(x_test_pad, y_test)

@app.route("/", methods=["GET", "POST"])
def adder_page():
    errors = ""
    if request.method == "POST":
        
        try:
            positive_review = (request.form["input"])
        except:
            errors += "<p>{!r} is not a number.</p>\n".format(request.form["input"])
        if positive_review is not None:
            text=[positive_review]
            tokens = tokenizer.texts_to_sequences(text)
            tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating='pre')
            a=model.predict(tokens_pad)
            
            return '''
                <html>
		<body style="background-color: #EAF0F1; text-align: center; margin: 15%;">
			<div>
    		<h2>Positive Review with a score of {} %</h2>
				<a href="/">Click here to calculate again</a>
			</div>
    </body>
</html>
            '''.format(a[0]*100)
    return '''
        <html>
    <body style="background-color: #EAF0F1; text-align: center; margin: 12%;">
      <div class="content">
        <div class="heading">
          <h1 style="text-align : center">
            SentimentAnalysis-RNN
          </h1>
        </div>
        <form method="post" action=".">
            <p>Enter your text: </p>
            <textarea name="input" rows="4" cols="40"></textarea><br><br>
            <input type="submit" value="Do calculation"
              style="background-color:#2ecc72;
                      border: none;
                      color: white;
                      padding: 15px 32px;
                      text-align: center;
                      text-decoration: none;
                      display: inline-block;
                      font-size: 16px;
                      box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19)" />
        </form>
      </div>
    </body>
</html>
    '''.format(errors=errors)
    
if __name__ == '__main__':
    app.run()




"""
if b > 0.50: # I am thresholding it at 50%
    print('Positive Review ')
else:
    print('Negative Review with a score of {} %'.format(b[0]*100))"""
