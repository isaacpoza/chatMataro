import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import logging
import json
import random
from keras.models import load_model
import tkinter
from tkinter import *

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicialización
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.keras')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Diccionario para manejar el contexto de los usuarios
contexto_usuario = {}
user_context = {}

def clean_up_sentence(sentence):
    logging.info(f"Lemmatizando y tokenizando la oración: {sentence}")
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    logging.info(f"Palabra encontrada en la bolsa: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    logging.info(f"Prediciendo la intención para el mensaje: {sentence}")
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0  # Ajusta este umbral según sea necesario
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        logging.info(f"Intención predicha: {classes[r[0]]} con probabilidad {r[1]}")
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse2(ints, intents_json, user_id):
    tag = ints[0]['intent']  # La intención predicha
    result = ''
    list_of_intents = intents_json['intents']
    current_context = contexto_usuario.get(user_id, '')

    # Verifica si la intención actual es válida para el contexto
    for i in list_of_intents:
        if i['tag'] == tag:
            # Maneja contexto de filtro
            if 'context_filter' in i:
                if isinstance(i['context_filter'], list):
                    if any(h in current_context for h in i['context_filter']):
                        result = random.choice(i['responses'])
                        break
                else:
                    if current_context == i['context_filter']:
                        result = random.choice(i['responses'])
                        break
            else:
                if 'context_set' in i:
                    logging.info(f"Ajustando el contexto del usuario {user_id} a: {i['context_set']}")
                    contexto_usuario[user_id] = i['context_set']
                if result == '':
                    result = random.choice(i['responses'])
    
    # Manejo especial para la intención "salida"
    if tag == "salida":
        logging.info(f"El usuario ha solicitado salir.")
        contexto_usuario[user_id] = ''  # Reinicia el contexto
        result = "De acuerdo, ¡hasta luego!"
    
    if result == '':
        if tag != "salida":
            result = "No estoy seguro de qué te refieres. ¿Podrías aclarar tu pregunta o intentar con otro tema?"
            if current_context:
                result += f" Antes hablamos sobre {current_context}. ¿Quieres continuar con ese tema?"

    logging.info(f"Respuesta seleccionada para la intención {tag}: {result}")
    return result

def chatbot_response(msg, user_id="default"):
    logging.info(f"Recibiendo mensaje del usuario {user_id}: {msg}")
    ints = predict_class(msg, model)

    if not ints:
        return "Lo siento, no entendí tu mensaje. ¿Puedes reformularlo o preguntarme otra cosa?"

    res = getResponse(ints, intents, user_id)
    return res

def cargar_descripcion(destino):
    try:
        with open(f"{destino}.txt", "r") as file:
            descripcion = file.read()
        return descripcion
    except FileNotFoundError:
        return "Lo siento, no tengo información sobre ese destino en este momento."

# Funciones de la interfaz gráfica
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        logging.info("Enviando mensaje al chatbot...")
        res = chatbot_response(msg, user_id="usuario_123")

        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

def reset_context():
    logging.info("Reiniciando el contexto del usuario 'usuario_123'.")
    contexto_usuario['usuario_123'] = ''
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "Bot: Contexto reiniciado.\n\n")
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)

# Crear GUI con tkinter
base = Tk()
base.title("Chatbot con Contexto y Logging")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height="5",
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

ResetButton = Button(base, font=("Verdana", 12, 'bold'), text="Reset Context", width="12", height="5",
                    bd=0, bg="#de3232", activebackground="#9d3c3c", fg='#ffffff',
                    command=reset_context)

scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=50, width=265)
SendButton.place(x=6, y=401, height=50)
ResetButton.place(x=6, y=460, height=30)

base.mainloop()
