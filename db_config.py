# db_config.py
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, desc, UniqueConstraint, Text, Date, Boolean, Numeric, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, time


# Crear la base declarativa
Base = declarative_base()

# Configurar la conexión a la base de datos (ajusta el string de conexión)
engine = create_engine("mysql+mysqlconnector://flask:flask2024@13.60.205.202/mataroin", echo=False)

# Crear la fábrica de sesiones
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Función para obtener la sesión en cualquier parte de la aplicación
def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
