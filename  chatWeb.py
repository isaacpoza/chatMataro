from flask import Flask, render_template, request, jsonify
import spacy
import random
import json
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import string
import cargacontenidos
from models import Chat
from db_config import SessionLocal

app = Flask(__name__)

# Cargar el modelo de SpaCy y la red neuronal
nlp = spacy.load('es_core_news_md')
model = load_model('chatbot_model.keras')

# Cargar las palabras y clases preprocesadas
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Cargar el archivo intents.json
with open('intents.json') as file:
    intents = json.load(file)

# Diccionario global para almacenar el contexto de los usuarios
user_context = {}

# Generar un usuario aleatorio para la sesión
caracteres = string.ascii_letters + string.digits
user_random = ''.join(random.choices(caracteres, k=6))

# Funciones del chatbot (similar a las funciones que ya tienes)
def clean_up_sentence(sentence):
    doc = nlp(sentence)
    lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return lemmas

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    if len(results) == 0:
        return []

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def getResponse(ints, intents_json, user_id):
    global user_context
    filtro = ''

    if len(ints) == 0:
        return "Lo siento, no puedo entender lo que dices. ¿Podrías reformular la pregunta?"

    tag = ints[0]['intent']
    probabilidad = float(ints[0]['probability'])

    if user_id not in user_context:
        user_context[user_id] = []

    user_previous_context = user_context[user_id]

    for i in intents_json['intents']:
        if i['tag'] == tag:
            if 'context_filter' in i and i['context_filter']:
                filtro = i['context_filter']

            if 'context_filter' in i and i['context_filter']:
                if not any(context in user_previous_context for context in i['context_filter']):
                    return "Lo siento, parece que primero debemos hablar de algo más."

            if 'context_set' in i:
                user_context[user_id] = i['context_set']

                if probabilidad >= 0.75:
                    return i['responses'][0]

            response = random.choice(i['responses'])
            return response
    return "Lo siento, no entiendo tu solicitud."

# Función para manejar la respuesta del chatbot
def chatbot_response(user_input, user_id):
    ints = predict_class(user_input)
    pueblo = cargacontenidos.hayPueblo(user_input)
    if pueblo is not None:
        return pueblo
    else:
        res = getResponse(ints, intents, user_id)
        return res

# Ruta principal para servir la página web
@app.route("/")
def index():
    return render_template("index.html")

# Ruta para manejar las solicitudes del chatbot (AJAX)
@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json.get("message")
    bot_response = chatbot_response(user_message, user_random)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
