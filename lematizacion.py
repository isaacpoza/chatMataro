import spacy
import json

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
        
        # Mostrar las palabras antes de la lematización
        original_words = [token.text for token in doc]
        print(f"Original words: {original_words}")
        
        # Lematizar las palabras y almacenarlas
        lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        
        # Mostrar las palabras después de la lematización
        print(f"Lemmatized words: {lemmas}\n")

        all_words.extend(lemmas)
        documents.append((lemmas, intent['tag']))

    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Eliminar duplicados y ordenar las palabras y las clases
all_words = sorted(set(all_words))
classes = sorted(set(classes))

