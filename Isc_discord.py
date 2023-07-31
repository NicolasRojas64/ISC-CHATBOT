import nltk
import pandas as pd
from nltk.stem.porter import *
stemmer = PorterStemmer()
import numpy
import tflearn
import tensorflow
import discord

nltk.download('punkt')
nltk.download('stopwords')

discord_key = "MTExMDE3OTI3NzI1MDUwMjY2Ng.GILDDm.aj9wv0fz2awSE5cONpi3jhnaP462DxO-zwfl6Y"
raw_data = pd.read_csv('Preguntas.csv',sep=',',encoding = "UTF-8")


from nltk.corpus import stopwords
stop = stopwords.words("spanish")

raw_data["pregunta"] = raw_data["pregunta"].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop]))

def tokenize(text):
  tokens = nltk.word_tokenize(text)
  return tokens

raw_data["pregunta"] = raw_data["pregunta"].apply(tokenize)

clean_data = raw_data

tags = []
for f in range(clean_data.index.stop):
  tags.append(f)

for f in range(len(tags)):
  clean_data["tag"] = tags[f]

for f in range(len(tags)):
  clean_data["tag"][f] = tags[f]

print(clean_data)

words = []

for question in clean_data["pregunta"]:
  for word in question:
    words.append(word)

print(words)

auxY = []
auxX = []
tags = []

c = 0
for question in clean_data["pregunta"]:
  auxX.append(question)
  auxY.append(clean_data["tag"][c])
  tags.append(clean_data["tag"][c])
  c+=1

print(auxX)
print(tags)

from nltk.stem.porter import *
stemmer = PorterStemmer()

words =  [stemmer.stem(w) for w in words]
words = sorted(list(set(words)))

training = []
output = []
emptyOutput = [0 for _ in range(len(tags))]

for x, data in enumerate(auxX):
  bucket = []
  auxWord = [stemmer.stem(w) for w in data]
  #auxWord = [lemma(data)]
  for w in words:
    if w in auxWord:
      bucket.append(1)
    else:
      bucket.append(0)
  outputRow = emptyOutput[:]
  outputRow[tags.index(auxY[x])] = 1
  training.append(bucket)
  output.append(outputRow)
print(training)
print(output)

training = numpy.array(training)
output =  numpy.array(output)

tensorflow.compat.v1.reset_default_graph()
network = tflearn.input_data(shape=[None, len(training[0])])
network = tflearn.fully_connected(network,20)
network = tflearn.fully_connected(network,20)
network = tflearn.fully_connected(network,len(output[0]),activation="softmax")
network = tflearn.regression(network)

model = tflearn.DNN(network)
model.fit(training, output, n_epoch=1500,batch_size=58,show_metric=True)
model.save("model.tflearn")


client = discord.Client(intents=discord.Intents.default())
@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.find("Hola")!=-1:
        await message.channel.send("Hola desde ISC_Chatbot, ¿En que puedo ayudarte?")
    
    else:
      bucket = [0 for _ in range(len(words))]

      processedInput = nltk.word_tokenize(message.content)
      print("Contenido: ", processedInput)
      processedInput = [stemmer.stem(w.lower()) for w in processedInput]
      for individualWord in processedInput:
          for i, word in enumerate(words):
              if word == individualWord:
                  bucket[i] = 1
      result = model.predict([numpy.array(bucket)])
      indexResults = numpy.argmax(result)
      tag = tags[indexResults]

      response = clean_data["respuesta"][tag]
      await message.channel.send(response)
  
client.run(discord_key)
