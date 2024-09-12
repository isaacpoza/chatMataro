import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
import string  # Para eliminar la puntuación
import re  # Para eliminar caracteres específicos usando expresiones regulares
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Crear el lematizador
lemmatizer = WordNetLemmatizer()

# Cargar las stopwords (en este caso en español)
stop_words = set(stopwords.words('spanish'))

# Función para mapear etiquetas POS a WordNet
def get_wordnet_pos(word):
    """Mapea la etiqueta POS de NLTK a la etiqueta POS de WordNet"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Función para limpiar caracteres no deseados como ¿, ¡, y otros signos de puntuación
def clean_text(word):
    # Eliminar caracteres como ¿ y ¡ de manera explícita
    word = word.replace('¿', '').replace('¡', '').replace('\'', '')
    
    # Alternativamente, podemos usar expresiones regulares para eliminar cualquier carácter no alfabético
    word = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ]', '', word)
    
    return word

# Inicializar variables
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',', '¿', '¡']  # Incluimos los signos de interrogación y exclamación

# Cargar el archivo intents.json
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Procesar el archivo intents.json
for intent in intents['intents']:
    for pattern in intent['patterns']:

        # Tokenizar cada palabra en el patrón
        w = nltk.word_tokenize(pattern)
        
        # Filtrar stopwords (palabras vacías) y eliminar signos de puntuación antes de lematizar
        w = [word for word in w if word not in stop_words and word not in ignore_words]
        
        # Limpiar cada palabra usando la función clean_text()
        w = [clean_text(word) for word in w]
        
        # Eliminar cualquier palabra vacía después de la limpieza
        w = [word for word in w if word != '']
        
        # Añadir las palabras filtradas a la lista de palabras
        words.extend(w)
        
        # Añadir los documentos (pareja de palabras e intención)
        documents.append((w, intent['tag']))

        # Añadir la clase (intención) si no está ya en la lista
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematización de las palabras restantes y eliminación de duplicados
words = [lemmatizer.lemmatize(w.lower(), get_wordnet_pos(w)) for w in words]
words = sorted(list(set(words)))

# Ordenar las clases
classes = sorted(list(set(classes)))

# Mostrar resultados
print(len(documents), "documentos")
print(len(classes), "clases", classes)
print(len(words), "palabras únicas lematizadas", words)

# Guardar palabras y clases en archivos pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Inicializar datos de entrenamiento
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # Inicializar la bolsa de palabras
    bag = []
    
    # Obtener las palabras tokenizadas del patrón
    pattern_words = doc[0]
    
    # Lematizar cada palabra en el patrón según su categoría gramatical
    pattern_words = [lemmatizer.lemmatize(word.lower(), get_wordnet_pos(word)) for word in pattern_words]
    
    # Crear la bolsa de palabras: 1 si la palabra está en el patrón, 0 si no
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # La salida es '0' para cada etiqueta y '1' para la etiqueta actual
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Barajar las características y convertir a np.array
random.shuffle(training)
training = np.array(training, dtype=object)

# Crear listas de entrenamiento. X - patrones, Y - intenciones
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Datos de entrenamiento creados")

# Crear el modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar el modelo con SGD
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Guardar el modelo
model.save('chatbot_model.keras', hist)

print("Modelo creado y guardado")
