from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, JSON, Date
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import date

Base = declarative_base()

class Dataset(Base):
    __tablename__ = 'dataset'
    dataset_id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)
    date_created = Column(Date, default=date.today)

class Molecule(Base):
    __tablename__ = 'molecule'
    molecule_id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('dataset.dataset_id'))
    name = Column(String)
    dataset = relationship("Dataset", back_populates="molecules")

class Conformer(Base):
    __tablename__ = 'conformer'
    conformer_id = Column(Integer, primary_key=True)
    molecule_id = Column(Integer, ForeignKey('molecule.molecule_id'))
    properties = Column(JSON)
    molecule = relationship("Molecule", back_populates="conformers")

class Fingerprint(Base):
    __tablename__ = 'fingerprint'
    fingerprint_id = Column(Integer, primary_key=True)
    conformer_id = Column(Integer, ForeignKey('conformer.conformer_id'))
    dataset_id = Column(Integer, ForeignKey('dataset.dataset_id'))
    name = Column(String)
    data = Column(JSON)
    dataset = relationship("Dataset", back_populates="fingerprints")
    conformer = relationship("Conformer", back_populates="fingerprints")

class Model(Base):
    __tablename__ = 'model'
    model_id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('dataset.dataset_id'))
    name = Column(String)
    hyperparameters = Column(JSON)
    dataset = relationship("Dataset", back_populates="models")

class PerformanceMetric(Base):
    __tablename__ = 'performance_metric'
    metric_id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('model.model_id'))
    fingerprint_id = Column(Integer, ForeignKey('fingerprint.fingerprint_id'))
    dataset_id = Column(Integer, ForeignKey('dataset.dataset_id'))
    ba = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    dataset = relationship("Dataset", back_populates="performance_metrics")
