import json

def hayPueblo(user_input):
#user_input = 'me gustaría ir a Mataró';

    datos={}
    with open('contenidos/pueblos.json', 'r') as archivo_json:
        datos = json.load(archivo_json)  # Cargar los datos como un diccionario de Python

    # Paso 2: Acceder a la lista de pueblos
    pueblos_lista = datos.get('pueblos', [])  # Obtener la lista de pueblos

    # Paso 3: Crear un diccionario donde las claves sean los nombres de los pueblos y los valores las descripciones
    pueblos_diccionario = {pueblo_info.get('pueblo'): pueblo_info.get('descripcion') for pueblo_info in pueblos_lista}

    # Paso 1: Convertir las claves del diccionario en un conjunto para una búsqueda rápida
    pueblos_claves = set(pueblos_diccionario.keys())  # Convertimos las claves en un conjunto


    # Paso 2: Buscar si alguna clave está contenida en la cadena
    for pueblo in pueblos_claves:
        if pueblo in user_input:
            # Paso 3: Usamos el método get para obtener la descripción de la clave encontrada
            return (pueblos_diccionario.get(pueblo))            
    return

print(hayPueblo('ir a Mataó'))

