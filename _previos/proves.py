import spacy

# Cargar el modelo de spaCy para español
nlp = spacy.load("es_core_news_md")

# Función para lematizar y procesar usando SpaCy
def lematizar_con_spacy(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

# Ejemplo de texto
texto = "'casa, pisos,casas'"
lemas = lematizar_con_spacy(texto)
print(lemas)  # Debería imprimir ['producto']
