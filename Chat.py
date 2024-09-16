import spacy
import random
import json
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import tkinter as tk
from models import Chat
from db_config import SessionLocal
from tkinter import *
import random
import string

# Definir los caracteres permitidos (letras mayúsculas, minúsculas y dígitos)
caracteres = string.ascii_letters + string.digits  # ascii_letters incluye mayúsculas y minúsculas

# Generar una cadena aleatoria de 6 caracteres
user_random = ''.join(random.choices(caracteres, k=6))

entrada=''
# Cargar el modelo de SpaCy en español
nlp = spacy.load('es_core_news_md')

# Cargar el modelo entrenado de la red neuronal
model = load_model('chatbot_model.keras')

# Cargar las palabras y clases preprocesadas con pickle
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Cargar el archivo intents.json
with open('intents.json') as file:
    intents = json.load(file)

# Diccionario global para almacenar el contexto de los usuarios
user_context = {}

# Función para lematizar una frase y devolver las palabras relevantes
def clean_up_sentence(sentence):
    doc = nlp(sentence)  # Procesa la frase con SpaCy
    lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return lemmas

# Convertir una frase en bolsa de palabras: 1 si la palabra está en el vocabulario, 0 si no
def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Función para predecir la clase (intención) del usuario
def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Si no hay resultados por encima del umbral, devolvemos una lista vacía
    if len(results) == 0:
        return []
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def check_variable(var_name):
    return f"{var_name}: {locals().get(var_name, f'{var_name} no existe')}"


    
# Función para obtener la respuesta adecuada basada en la intención y el contexto del usuario
def getResponse(ints, intents_json, user_id):
    global user_context
    filtro=''

    if len(ints) == 0:  # Verificar si no hay intenciones predichas
        return "Lo siento, no puedo entender lo que dices. ¿Podrías reformular la pregunta?"
    
    tag = ints[0]['intent']  # Intención más probable
    probabilidad = float(ints[0]['probability'])  # Obtener la probabilidad asociada

    # Si el user_id no existe en user_context, inicializar con una lista vacía
    if user_id not in user_context:
        user_context[user_id] = []

    user_previous_context = user_context[user_id]  # Verificar el contexto actual del usuario

    # Recorrer las intenciones en el archivo intents.json
    for i in intents_json['intents']:
        if i['tag'] == tag:
            if 'context_filter' in i and i['context_filter']:
                filtro=i['context_filter']
            
            insertChat(user_id,entrada,str(user_context[user_id]),filtro)

            # Verificar si la intención tiene un context_filter y si coincide con algún contexto del usuario
            if 'context_filter' in i and i['context_filter']: 
                if not any(context in user_previous_context for context in i['context_filter']) :
                    print(check_variable('tag'), check_variable('user_context[user_id]'), check_variable('user_previous_context'))

                    return "Lo siento, parece que primero debemos hablar de algo más."
                else:
                    if user_previous_context not in tag and tag != 'reformulacion':
                                print(check_variable('tag'), check_variable('user_context[user_id]'), check_variable('user_previous_context'))
                                return "Cambiar tema."
                    else:

                        return random.choice(i['responses'])#"Lo siento, parece que primero debemos hablar de algo más."

            # Si la intención tiene un context_set y coincide con alguno de los context_filter, seleccionar la respuesta basada en la probabilidad
            if 'context_set' in i:
                user_context[user_id] = i['context_set'] # Añadir contextos sin duplicados
             
                # Si el contexto coincide con el filtro y la probabilidad es alta, seleccionar la respuesta más probable
                if 'context_filter' in i and any(context in i['context_filter'] for context in i['context_set']):
                    if probabilidad >= 0.75:  # Umbral de alta probabilidad
                        print(check_variable('tag'), check_variable('user_context[user_id]'), check_variable('user_previous_context'))
                        return i['responses'][0]  # Seleccionar la primera respuesta como la más probable

            # Seleccionar una respuesta aleatoria si no se cumplen las condiciones anteriores
            response = random.choice(i['responses'])
            return response
    return "Lo siento, no entiendo tu solicitud."


# Función para restablecer el contexto de un usuario
def reset_user_context(user_id):
    if user_id in user_context:
        del user_context[user_id]
        print(f"Contexto del usuario {user_id} restablecido.")

# Función principal para manejar la respuesta del chatbot
def chatbot_response(user_input, user_id):
    ints = predict_class(user_input)  # Predecir la intención del usuario
    
    if "París" in user_input or "paris" in user_input:
    # Si el usuario pregunta sobre París, se carga el archivo paris.txt
        return cargar_descripcion("paris")
    else:
        res = getResponse(ints, intents, user_id)
    return res
    
    res = getResponse(ints, intents, user_id)  # Obtener la respuesta basada en la intención y el contexto
    return res

def cargar_descripcion(destino):
    try:
        with open(f"contenidos/{destino}.txt", "r") as file:
            descripcion = file.read()
        return descripcion
    except FileNotFoundError:
        return "Lo siento, no tengo información sobre ese destino en este momento."


#---------BBDD
def insertChat(usuario, consulta, context_set, context_filter):
    # Obtener la sesión de la base de datos
    db = SessionLocal()

    try:
        # Crear una nueva instancia de la tabla Chat
        nuevo_chat = Chat(
            usuario=usuario,
            consulta=consulta,
            context_set="".join(context_set),
            context_filter="".join(context_set)
        )

        # Agregar el nuevo registro a la sesión
        db.add(nuevo_chat)

        # Confirmar la transacción
        db.commit()

        # Refrescar el objeto para obtener el ID generado
        db.refresh(nuevo_chat)

        print(f"Inserción exitosa con ID: {nuevo_chat.id}")
        
    except Exception as e:
        # Si ocurre un error, deshacer la transacción
        db.rollback()
        print(f"Error en la inserción: {e}")
    
    finally:
        # Cerrar la sesión
        db.close()


# ---- INTERFAZ GRÁFICA CON TKINTER ----

# Crear la ventana principal
root = Tk()
root.title("Chatbot con Tkinter y Contexto")
root.geometry("400x500")

# Crear el cuadro de texto donde se mostrarán las conversaciones
chat_log = Text(root, bg="white", height="8", width="50", font=("Arial", 12))
chat_log.config(state=DISABLED)

# Crear una barra de desplazamiento para el cuadro de texto
scrollbar = Scrollbar(root, command=chat_log.yview)
chat_log['yscrollcommand'] = scrollbar.set

# Crear el campo de entrada de texto
entry_box = Entry(root, width="29", font=("Arial", 12))

# Función que maneja la interacción cuando el usuario envía un mensaje
def send():
    global entrada
    user_message = entry_box.get()  # Obtener el mensaje del usuario
    entry_box.delete(0, END)  # Limpiar el campo de entrada
    entrada=user_message
    if user_message.lower() == "salir":
        reset_user_context(user_random)  # Restablecer el contexto si el usuario finaliza
        chat_log.config(state=NORMAL)
        chat_log.insert(END, "Bot: ¡Adiós!\n\n")
        chat_log.config(state=DISABLED)
        return
    
    # Mostrar el mensaje del usuario en el cuadro de texto
    chat_log.config(state=NORMAL)
    chat_log.insert(END, f"Tú: {user_message}\n\n")
    chat_log.config(state=DISABLED)
    
    # Obtener la respuesta del chatbot
    bot_response = chatbot_response(user_message, user_random)
    
    # Mostrar la respuesta del bot en el cuadro de texto
    chat_log.config(state=NORMAL)
    chat_log.insert(END, f"Bot: {bot_response}\n\n")
    chat_log.config(state=DISABLED)
    
    # Desplazar el cuadro de texto hacia abajo
    chat_log.yview(END)

# Botón para enviar el mensaje
send_button = Button(root, text="Enviar", width="12", height=1, bg="blue", fg="white", command=send)

# Posicionar los elementos en la ventana
chat_log.grid(row=0, column=0, columnspan=2)
scrollbar.grid(row=0, column=2, sticky='ns')
entry_box.grid(row=1, column=0)
send_button.grid(row=1, column=1)

# Iniciar la interfaz gráfica
root.mainloop()
