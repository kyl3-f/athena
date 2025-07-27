# src/models/market_data.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class StockPrice(Base):
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )

class OptionsData(Base):
    __tablename__ = 'options_data'
    
    id = Column(Integer, primary_key=True)
    underlying_symbol = Column(String(10), nullable=False)
    contract_type = Column(String(4), nullable=False)  # CALL or PUT
    strike_price = Column(Float, nullable=False)
    expiration_date = Column(DateTime, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    bid = Column(Float)
    ask = Column(Float)
    last_price = Column(Float)
    volume = Column(Integer)
    open_interest = Column(Integer)
    implied_volatility = Column(Float)
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_underlying_timestamp', 'underlying_symbol', 'timestamp'),
        Index('idx_contract_details', 'underlying_symbol', 'contract_type', 'strike_price', 'expiration_date'),
    )