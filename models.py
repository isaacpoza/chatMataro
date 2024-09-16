from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, desc, UniqueConstraint, Text, Date, Boolean, Numeric, DECIMAL
from db_config import Base
from datetime import datetime
from pytz import timezone

def get_current_time_in_spain():
    """
    Obtiene la hora actual en la zona horaria de España.

    :return: La hora actual en la zona horaria de España.
    """
    spain_tz = pytz.timezone('Europe/Madrid')
    return datetime.now(spain_tz)

# Configuración de la conexión a la base de datos
#engine = create_engine("mysql+mysqlconnector://flask:flask2024@localhost/mataroin", echo=False)
#engine = create_engine("mysql+mysqlconnector://flask:flask2024@13.60.205.202/mataroin", echo=False)

# Declarative base

class Alojamientos(Base):
    """
    Modelo de datos para la tabla 'alojamientos'.
    """
    __tablename__ = 'alojamientos'

    id = Column(Integer, primary_key=True)
    id_localidad = Column(Integer)
    id_proveedor = Column(Integer)
    tipo = Column(String(50))
    nombre = Column(String(100))
    direccion = Column(String(100))
    telefono = Column(String(45))
    correo = Column(String(100))
    id_imagen = Column(Integer)
    descripcion = Column(String(200))
    f_crea = Column(DateTime)
    f_mod = Column(DateTime)
    f_elim = Column(DateTime)

class Destino(Base):
    """
    Modelo de datos para la tabla 'destinos'.
    """
    __tablename__ = 'destinos'

    id = Column(Integer, primary_key=True, autoincrement=True)
    id_localidad = Column(Integer, nullable=True)
    id_alojamiento = Column(Integer, nullable=True)
    temporada_verano = Column(Float, nullable=True)
    temporada_primavera = Column(Float, nullable=True)
    temporada_especial = Column(Float, nullable=True)
    tarjeta_oro = Column(Float, nullable=True)
    tarjeta_plata = Column(Float, nullable=True)
    dextras_verano = Column(Float, nullable=True)
    dextras_primavera = Column(Float, nullable=True)
    dextras_resto = Column(Float, nullable=True)
    resultado = Column(Float, nullable=False)
    f_crea = Column(DateTime, nullable=True)
    f_mod = Column(DateTime, nullable=True)
    f_elim = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<alojamientos(id='{self.id}', idlocalidiad={self.id_localidad} id:alojamiento{self.id_alojamiento})>"

class Dispositivos(Base):
    """
    Modelo de datos para la tabla 'dispositivos'.
    """
    __tablename__ = 'dispositivos'
    
    id = Column(Integer, primary_key=True)
    nombre = Column(String(25), nullable=False)
    descripcion = Column(String(240))
    f_crea = Column(DateTime)
    f_mod = Column(DateTime)
    f_elim = Column(DateTime)

    __table_args__ = (
        UniqueConstraint('id'),
    )

    def __repr__(self):
        return f"<Dispositivos(id='{self.id}', nombre='{self.nombre}')>"

class Paises(Base):
    """
    Modelo de datos para la tabla 'paises'.
    """
    __tablename__ = 'paises'
    
    id = Column(Integer, primary_key=True)
    nombre = Column(String(50))
    f_crea = Column(DateTime, default=datetime.now)
    f_mod = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    f_elim = Column(DateTime)

    __table_args__ = (
        UniqueConstraint('id'),
    )

    def __repr__(self):
        return f"<Paises(id='{self.id}', nombre='{self.nombre}')>"

class Tags(Base):
    """
    Modelo de datos para la tabla 'tags'.
    """
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String(50), nullable=False)
    descripcion = Column(String(240), nullable=False)
    f_crea = Column(Date)
    f_mod = Column(Date)
    f_elim = Column(Date)

    __table_args__ = (
        UniqueConstraint('id'),
    )

    def __repr__(self):
        return f"<Tags(id='{self.id}', nombre='{self.nombre}')>"

class Cookies(Base):
    """
    Modelo de datos para la tabla 'cookies'.
    """
    __tablename__ = 'cookies'
    
    id = Column(Integer, primary_key=True)
    id_pais = Column(Integer)
    id_dispositivo = Column(Integer)
    fecha_y_hora = Column(DateTime)
    id_tag = Column(Integer)
    comprar = Column(Boolean)
    f_crea = Column(DateTime)
    f_mod = Column(DateTime)
    f_elim = Column(DateTime)

    def __repr__(self):
        return f"<Cookies(id='{self.id}', fecha_y_hora='{self.fecha_y_hora}')>"

class Estadisticas(Base):
    """
    Modelo de datos para la tabla 'estadisticas'.
    """
    __tablename__ = 'estadisticas'
    id = Column(Integer, primary_key=True, autoincrement=True)
    descripcion = Column(String(200))
    estadistica = Column(Text)
    f_crea = Column(DateTime, default=None)
    f_mod = Column(DateTime, default=None)
    f_elim = Column(DateTime, default=None)

class Biblioteca(Base):
    """
    Modelo de datos para la tabla 'biblioteca'.
    """
    __tablename__ = 'biblioteca'

    id = Column(Integer, primary_key=True)
    tipo = Column(String(50))
    nombre = Column(String(100))
    descripcion = Column(String(50))
    filename = Column(String(50))
    duracion = Column(Integer)
    texto = Column(String(50))
    imagen = Column(String(50))
    formato = Column(String(50))
    f_crea = Column(DateTime, default=get_current_time_in_spain)
    f_mod = Column(DateTime, default=get_current_time_in_spain, onupdate=get_current_time_in_spain)

class TagsRelacion(Base):
    """
    Modelo de datos para la tabla 'tags_relacion'.
    """
    __tablename__ = 'tags_relacion'

    id = Column(Integer, primary_key=True, autoincrement=True)
    origen = Column(String, nullable=False)
    id_origen = Column(Integer, nullable=False)
    id_tag = Column(Integer, nullable=False)
    f_crea = Column(DateTime, default=get_current_time_in_spain)
    f_mod = Column(DateTime, default=get_current_time_in_spain, onupdate=get_current_time_in_spain)
    f_elim = Column(DateTime, default=get_current_time_in_spain)

    def __repr__(self):
        return f"<Biblioteca(id='{self.id}', tipo={self.tipo}, nombre={self.nombre}, descripcion={self.descripcion})>"

    def guardar(self, session):
        """
        Guarda la instancia actual en la sesión de la base de datos.

        :param session: La sesión de la base de datos.
        """
        session.add(self)
        session.commit()

class UsabDatos(Base):
    """
    Modelo de datos para la tabla 'usab_datos'.
    """
    __tablename__ = 'usab_datos'
    id = Column(Integer, primary_key=True)
    origen = Column(String(255))
    id_origen = Column(Integer)
    id_usab = Column(Integer)
    media = Column(Float)
    mediana = Column(Float)
    moda = Column(Float)

class UsabRelacion(Base):
    """
    Modelo de datos para la tabla 'usab_relacion'.
    """
    __tablename__ = 'usab_relacion'
    id = Column(Integer, primary_key=True)
    id_alojamientos = Column(Integer)
    id_usab = Column(Integer)
    f_crea = Column(DateTime)
    f_mod = Column(DateTime)
    valor = Column(Float)

class Localidades(Base):
    """
    Modelo de datos para la tabla 'localidades'.
    """
    __tablename__ = 'localidades'

    id = Column(Integer, primary_key=True)
    nombre = Column(String(100))
    poblacion = Column(Integer)
    f_crea = Column(DateTime)
    f_mod = Column(DateTime)
    f_elim = Column(DateTime)

    def __repr__(self):
        return f"<localidad(id='{self.id}', nombre={self.nombre}, poblacion={self.poblacion})>"

class Hoteles(Base):
    """
    Modelo de datos para la tabla 'hoteles'.
    """
    __tablename__ = 'hoteles'

    id = Column(Integer, ForeignKey('alojamientos.id'), primary_key=True)
    estrellas = Column(String(45))
    habitaciones = Column(String(45))
    suites = Column(String(45))
    f_crea = Column(DateTime)
    f_mod = Column(DateTime)
    f_elim = Column(DateTime)

    def __repr__(self):
        return f"<hotel(id='{self.id}', numero de estrellas={self.estrellas})>"

class Campings(Base):
    """
    Modelo de datos para la tabla 'camping'.
    """
    __tablename__ = 'camping'

    id = Column(Integer, ForeignKey('alojamientos.id'), primary_key=True)
    tipo_camping = Column(String(45))
    plazas = Column(Integer)
    f_crea = Column(DateTime)
    f_mod = Column(DateTime)
    f_elim = Column(DateTime)

    def __repr__(self):
        return f"<camping(id='{self.id}', tipo_camping={self.tipo_camping})>"

class Hostales(Base):
    """
    Modelo de datos para la tabla 'hostales'.
    """
    __tablename__ = 'hostales'

    id = Column(Integer, ForeignKey('alojamientos.id'), primary_key=True)
    categoria = Column(String(45))
    habitaciones = Column(String(45))
    f_crea = Column(DateTime)
    f_mod = Column(DateTime)
    f_elim = Column(DateTime)

    def __repr__(self):
        return f"<hostales(id='{self.id}', categoria={self.categoria})>"

class Apartamentos(Base):
    """
    Modelo de datos para la tabla 'apartamentos'.
    """
    __tablename__ = 'apartamentos'

    id = Column(Integer, ForeignKey('alojamientos.id'), primary_key=True)
    habitaciones = Column(String(45))
    f_crea = Column(DateTime)
    f_mod = Column(DateTime)
    f_elim = Column(DateTime)

    def __repr__(self):
        return f"<apartamentos(id='{self.id}')>"

class Servicios(Base):
    """
    Modelo de datos para la tabla 'servicios'.
    """
    __tablename__ = 'servicios'

    id = Column(Integer, primary_key=True)
    nombre = Column(String(100))
    descripcion = Column(String(250))
    img_servicio = Column(String(50))
    f_crea = Column(DateTime)
    f_mod = Column(DateTime)

    def __repr__(self):
        return f"<Los servicios disponibles para este alojamiento son: '{self.nombre}')>"

class Alojamientos_servicios(Base):
    """
    Modelo de datos para la tabla 'alojamientos_servicios'.
    """
    __tablename__ = 'alojamientos_servicios'

    id = Column(Integer, primary_key=True)
    id_alojamientos = Column(Integer, ForeignKey('alojamientos.id'))
    id_servicios = Column(Integer, ForeignKey('servicios.id'))
    f_crea = Column(DateTime)
    f_mod = Column(DateTime)
    f_elim = Column(DateTime)

class Alojamientos_fechas(Base):
    """
    Modelo de datos para la tabla 'alojamientos_fechas'.
    """
    __tablename__ = 'alojamientos_fechas'
    id = Column(Integer, ForeignKey('alojamientos.id'), primary_key=True)
    id_alojamiento = Column(Integer, ForeignKey('alojamientos.id'))
    mes = Column(String(45))
    anyo = Column(String(45))
    fechasMes = Column(String(45))
    libresMes = Column(String(45))
    f_crea = Column(DateTime)
    f_mod = Column(DateTime)
    f_elim = Column(DateTime)
    precios = Column(Numeric(10, 2))

    def __repr__(self):
        return f" Se dispone de {self.libresMes} habitaciones libres para la fecha {self.fechasMes} de {self.mes} y su precio es {self.precios}>"

class FechasAlojamientos(Base):
    """
    Modelo de datos para la tabla 'fechas_alojamientos'.
    """
    __tablename__ = 'fechas_alojamientos'

    id = Column(Integer, primary_key=True, autoincrement=True)
    id_alojamiento = Column(Integer, nullable=False)
    fecha = Column(Date, nullable=False)
    libres = Column(Integer, nullable=False)
    disponible = Column(Boolean, nullable=False, default=True)
    precio = Column(DECIMAL(10, 2))
    f_crea = Column(DateTime, nullable=False, default=datetime.now())
    f_mod = Column(DateTime, default=None, onupdate=datetime.now())
    f_elim = Column(DateTime, default=None)

    def __repr__(self):
        return (f"<FechasAlojamientos(id={self.id}, id_alojamiento={self.id_alojamiento}, "
                f"fecha={self.fecha}, libres={self.libres}, disponible={self.disponible}, "
                f"precio={self.precio}, f_crea={self.f_crea}, f_mod={self.f_mod}, f_elim={self.f_elim})>")


# Definir el modelo de la tabla 'Chat'
class Chat(Base):
    __tablename__ = 'chat'

    id = Column(Integer, primary_key=True, autoincrement=True)
    usuario = Column(String(45))
    consulta = Column(Text)
    context_set = Column(String(45))
    context_filter = Column(String(45))


