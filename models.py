from sqlalchemy import Column, Integer, Float, String, DateTime
from database import Base


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    timestamp = Column(DateTime)

    # Thermal sensor
    object_temperature = Column(Float)
    ambient_temperature = Column(Float)

    # Gyro / accelerometer module
    vibration = Column(Float)
    alignment = Column(Float)

    # Load cell
    load = Column(Float)

    # Motor speed
    speed = Column(Float)

    # Flame sensor
    # 1 = safe
    # 0 = flame detected
    flame = Column(Integer)

    # ML + system result
    prediction = Column(String)
    confidence = Column(Float)
    severity = Column(String)
    health_score = Column(Float)