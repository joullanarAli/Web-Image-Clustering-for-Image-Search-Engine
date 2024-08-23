from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///./app_database.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()  # Updated import path

class User(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    #created_at = Column(DateTime, default=datetime.utcnow)

class Role(Base):
    __tablename__ = 'roles'
    role_id = Column(Integer, primary_key=True, index=True)
    role = Column(String, unique=True, index=True, nullable=False)


class UserRole(Base):
    __tablename__ = 'userRoles'
    user_role_id =Column(Integer, primary_key=True, index=True)
    role_id = Column(Integer, ForeignKey('roles.role_id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    user = relationship("User")
    role = relationship("role")

class SearchHistory(Base):
    __tablename__ = 'search_history'
    search_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    search_query = Column(Text, nullable=False)
    # search_date = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User")

# Create the tables
Base.metadata.create_all(bind=engine)
