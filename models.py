from sqlalchemy import Column, Integer, String , DateTime  , Text , Boolean , ForeignKey
from sqlalchemy.sql import func
from database import Base
from sqlalchemy.orm import relationship
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    contact = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), default= datetime.now())

    user_queries = relationship("UserQuery", back_populates="user")
    

class UserQuery(Base):
    __tablename__ = "user_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # Added ForeignKey
    query_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default= datetime.now())
    updated_at = Column(DateTime(timezone=True), default= datetime.now())

    # ✅ Add this line
    user = relationship("User", back_populates="user_queries")