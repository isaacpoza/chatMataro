import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import spacy
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import pickle

# Cargar el modelo de SpaCy en español
nlp = spacy.load('es_core_news_md')

# Cargar los datos del archivo intents.json
with open('intents.json') as file:
    data = json.load(file)

# Lista para almacenar las palabras lematizadas, etiquetas y documentos
all_words = []
classes = []
documents = []

# Iterar sobre cada intención en el archivo intents.json
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Procesar el texto con SpaCy
        doc = nlp(pattern)
        
        # Extraer las palabras lematizadas y almacenarlas
        lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        all_words.extend(lemmas)
        documents.append((lemmas, intent['tag']))

    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Eliminar duplicados y ordenar las palabras y las clases
all_words = sorted(set(all_words))
classes = sorted(set(classes))

# Crear los datos de entrenamiento
training = []
output_empty = [0] * len(classes)

# Crear el conjunto de entrenamiento con los datos lematizados
for doc in documents:
    bag = []
    pattern_words = doc[0]
    
    # Crear el "bag of words" con las palabras lematizadas
    bag = [1 if word in pattern_words else 0 for word in all_words]
    
    # Crear el vector de salida
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])


# Convertir el conjunto de entrenamiento a arrays de NumPy
training = np.array(training, dtype=object)

# Separar las características y las etiquetas
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

# Guardar las palabras y clases preprocesadas con pickle
with open("words.pkl", "wb") as f:
    pickle.dump(all_words, f)

with open("classes.pkl", "wb") as f:
    pickle.dump(classes, f)

# Crear el modelo de la red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
hist = model.fit(train_x, train_y, epochs=500, batch_size=5, verbose=1)

# Guardar el modelo entrenado
model.save("chatbot_model.keras", hist)

print('modelo_creado')
