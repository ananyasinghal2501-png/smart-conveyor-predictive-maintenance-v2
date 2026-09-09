from sqlalchemy import Column, Integer, Float, String, DateTime
from database import Base


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)

    timestamp = Column(DateTime)

    temperature = Column(Float)
    vibration = Column(Float)
    gyro = Column(Float)
    flame = Column(Integer)

    prediction = Column(String)
    confidence = Column(Float)
    severity = Column(String)
    health_score = Column(Float)