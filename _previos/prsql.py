# insert_chat.py

from models import Chat
from db_config import SessionLocal

def insertar_chat(usuario, consulta, context_set, context_filter):
    # Obtener la sesión de la base de datos
    db = SessionLocal()

    try:
        # Crear una nueva instancia de la tabla Chat
        nuevo_chat = Chat(
            usuario=usuario,
            consulta=consulta,
            context_set=context_set,
            context_filter=context_filter
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

# Ejemplo de uso
if __name__ == "__main__":
    insertar_chat("usuario1", "¿Cómo usar SQLAlchemy?", "general", "faq")
