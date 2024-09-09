#ejercicio --> conseguir por programación que no se repitan las preguntas
#cada vez que se responda a una pregunta del intent no se tendrá en cuenta en el resto dela conversación

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import logging

from keras.models import load_model
model = load_model('chatbot_model.keras')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))




# Diccionario para manejar el contexto de los usuarios
contexto_usuario = {}

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_up_sentence(sentence):
    logging.info(f"Lemmatizando y tokenizando la oración: {sentence}")
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
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
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        logging.info(f"Intención predicha: {classes[r[0]]} con probabilidad {r[1]}")
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json, user_id):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    logging.info(f"context_set{contexto_usuario.get(user_id, '')}")
             
    for i in list_of_intents:
        if i['tag'] == tag:
            if 'context_filter' in i and i['context_filter'] in contexto_usuario.get(user_id, ''):
                result = random.choice(i['responses'])
                break
            if 'context_set' in i:
                logging.info(f"Ajustando el contexto del usuario {user_id} a: {i['context_set']}")
                contexto_usuario[user_id] = i['context_set']
            else:
                contexto_usuario[user_id] = ''  # Si no hay contexto, lo vaciamos
            result = random.choice(i['responses'])
            break
    logging.info(f"Respuesta seleccionada para la intención {tag}: {result}")
    return result

def chatbot_response(msg, user_id="default"):
    logging.info(f"Recibiendo mensaje del usuario {user_id}: {msg}")
    ints = predict_class(msg, model)

    if "París" in msg or "paris" in msg:
        # Si el usuario pregunta sobre París, se carga el archivo paris.txt
        return cargar_descripcion("paris")
    else:
        res = getResponse(ints, intents, user_id)
    return res


def cargar_descripcion(destino):
    # Cargar la descripción desde un archivo de texto
    try:
        with open(f"{destino}.txt", "r") as file:
            descripcion = file.read()
        return descripcion
    except FileNotFoundError:
        return "Lo siento, no tengo información sobre ese destino en este momento."


#Creating GUI with tkinter
import tkinter
from tkinter import *

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

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

base = Tk()
base.title("Chatbot con Contexto y Logging")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")

# Botón para reiniciar contexto
ResetButton = Button(base, font=("Verdana",12,'bold'), text="Reset Context", width="12", height=5,
                    bd=0, bg="#de3232", activebackground="#9d3c3c",fg='#ffffff',
                    command= reset_context)

#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=50, width=265)
SendButton.place(x=6, y=401, height=50)
ResetButton.place(x=6, y=460, height=30)

base.mainloop()
