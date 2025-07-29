#!/usr/bin/env python3
"""
ML Signal Generator for Athena Trading System
Trains models and generates trading signals from options/stock features
"""

import logging
import polars as pl
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal container"""
    symbol: str
    timestamp: datetime
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1
    signal_strength: float  # 0-10
    features_used: Dict[str, float]
    model_probabilities: Dict[str, float]
    reasoning: str


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_val_score: float
    feature_importance: Dict[str, float]


class MLSignalGenerator:
    """ML-powered trading signal generator"""
    
    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.feature_columns = []
        
    def load_training_data(self, lookback_days: int = 30) -> pl.DataFrame:
        """Load historical ML features for training"""
        logger.info(f"Loading training data for last {lookback_days} days")
        
        # Look for ML feature files
        feature_files = list(self.data_dir.glob("**/ml_ready/ml_features_*.parquet"))
        
        if not feature_files:
            logger.error("No ML feature files found")
            return pl.DataFrame()
        
        # Load recent files
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_files = [
            f for f in feature_files 
            if datetime.fromtimestamp(f.stat().st_mtime) >= cutoff_date
        ]
        
        if not recent_files:
            logger.warning(f"No recent feature files found, using all {len(feature_files)} files")
            recent_files = feature_files
        
        # Combine all data
        all_data = []
        for file_path in recent_files:
            try:
                df = pl.read_parquet(file_path)
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        if all_data:
            combined_df = pl.concat(all_data)
            logger.info(f"Loaded {combined_df.shape[0]} training samples from {len(all_data)} files")
            return combined_df
        else:
            logger.error("No training data loaded")
            return pl.DataFrame()
    
    def create_target_labels(self, features_df: pl.DataFrame, 
                           future_return_threshold: float = 0.02) -> pl.DataFrame:
        """Create target labels for supervised learning"""
        if features_df.is_empty():
            return features_df
        
        logger.info(f"Creating target labels with {future_return_threshold:.1%} threshold")
        
        # Sort by symbol and timestamp
        df = features_df.sort(['symbol', 'timestamp'])
        
        # Calculate future returns (simplified - using price change as proxy)
        # In production, you'd want to look ahead to actual future prices
        df = df.with_columns([
            # Create target based on current price momentum and options flow
            pl.when(
                (pl.col('price_change_pct') > future_return_threshold * 100) &
                (pl.col('call_put_ratio') > 1.2) &
                (pl.col('net_gamma_exposure') > 0)
            ).then(2)  # BUY signal
            .when(
                (pl.col('price_change_pct') < -future_return_threshold * 100) &
                (pl.col('call_put_ratio') < 0.8) &
                (pl.col('net_gamma_exposure') < 0)
            ).then(0)  # SELL signal
            .otherwise(1)  # HOLD signal
            .alias('target')
        ])
        
        # Add target label names
        df = df.with_columns([
            pl.when(pl.col('target') == 2).then('BUY')
            .when(pl.col('target') == 0).then('SELL')
            .otherwise('HOLD')
            .alias('target_label')
        ])
        
        # Show distribution
        target_dist = df.group_by('target_label').agg(pl.count().alias('count'))
        logger.info("Target distribution:")
        for row in target_dist.iter_rows(named=True):
            logger.info(f"  {row['target_label']}: {row['count']}")
        
        return df
    
    def prepare_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for ML training"""
        if df.is_empty():
            return np.array([]), np.array([]), []
        
        # Define feature columns (exclude metadata and target)
        exclude_cols = ['symbol', 'timestamp', 'target', 'target_label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
        
        # Extract features and targets
        X = df.select(feature_cols).to_numpy()
        y = df.select('target').to_numpy().flatten()
        
        # Handle any NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {y.shape}")
        
        return X, y, feature_cols
    
    def train_models(self, features_df: pl.DataFrame) -> Dict[str, ModelPerformance]:
        """Train all ML models"""
        logger.info("Starting model training...")
        
        # Create targets
        df_with_targets = self.create_target_labels(features_df)
        
        if df_with_targets.is_empty():
            logger.error("No training data with targets")
            return {}
        
        # Prepare features
        X, y, feature_cols = self.prepare_features(df_with_targets)
        
        if X.size == 0:
            logger.error("No feature data prepared")
            return {}
        
        self.feature_columns = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        performances = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Use scaled features for logistic regression, raw for tree-based
                if model_name == 'logistic':
                    X_train_model = X_train_scaled
                    X_test_model = X_test_scaled
                else:
                    X_train_model = X_train
                    X_test_model = X_test
                
                # Train model
                model.fit(X_train_model, y_train)
                
                # Evaluate
                train_score = model.score(X_train_model, y_train)
                test_score = model.score(X_test_model, y_test)
                
                # Cross validation
                cv_scores = cross_val_score(
                    model, X_train_model, y_train, cv=5, scoring='accuracy'
                )
                
                # Predictions for detailed metrics
                y_pred = model.predict(X_test_model)
                
                # Feature importance (if available)
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(feature_cols, model.feature_importances_))
                    # Sort by importance
                    feature_importance = dict(
                        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    )
                elif hasattr(model, 'coef_'):
                    # For logistic regression
                    coef_abs = np.abs(model.coef_[0])
                    importance_dict = dict(zip(feature_cols, coef_abs))
                    feature_importance = dict(
                        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    )
                
                # Store performance
                performances[model_name] = ModelPerformance(
                    accuracy=test_score,
                    precision=test_score,  # Simplified for multi-class
                    recall=test_score,
                    f1_score=test_score,
                    cross_val_score=cv_scores.mean(),
                    feature_importance=feature_importance
                )
                
                # Store trained model
                self.trained_models[model_name] = model
                
                logger.info(f"  {model_name} - Accuracy: {test_score:.3f}, CV: {cv_scores.mean():.3f}")
                
                # Show top features
                if feature_importance:
                    top_features = list(feature_importance.keys())[:5]
                    logger.info(f"  Top features: {top_features}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Save models and scaler
        self.save_models()
        
        return performances
    
    def save_models(self):
        """Save trained models and scaler"""
        logger.info("Saving trained models...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save each model
        for model_name, model in self.trained_models.items():
            model_file = self.model_dir / f"{model_name}_{timestamp}.joblib"
            joblib.dump(model, model_file)
            
            # Also save as latest
            latest_file = self.model_dir / f"{model_name}_latest.joblib"
            joblib.dump(model, latest_file)
        
        # Save scaler
        scaler_file = self.model_dir / f"scaler_{timestamp}.joblib"
        joblib.dump(self.scaler, scaler_file)
        
        latest_scaler = self.model_dir / "scaler_latest.joblib"
        joblib.dump(self.scaler, latest_scaler)
        
        # Save feature columns
        feature_file = self.model_dir / "feature_columns.joblib"
        joblib.dump(self.feature_columns, feature_file)
        
        logger.info(f"Models saved to {self.model_dir}")
    
    def load_models(self):
        """Load trained models"""
        logger.info("Loading trained models...")
        
        try:
            # Load feature columns
            feature_file = self.model_dir / "feature_columns.joblib"
            if feature_file.exists():
                self.feature_columns = joblib.load(feature_file)
            
            # Load scaler
            scaler_file = self.model_dir / "scaler_latest.joblib"
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
            
            # Load models
            for model_name in self.models.keys():
                model_file = self.model_dir / f"{model_name}_latest.joblib"
                if model_file.exists():
                    self.trained_models[model_name] = joblib.load(model_file)
                    logger.info(f"Loaded {model_name}")
            
            logger.info(f"Loaded {len(self.trained_models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def generate_signal(self, features_df: pl.DataFrame) -> List[TradingSignal]:
        """Generate trading signals for given features"""
        if features_df.is_empty() or not self.trained_models:
            logger.warning("No features or trained models available")
            return []
        
        logger.info(f"Generating signals for {features_df.shape[0]} symbols")
        
        # Prepare features
        exclude_cols = ['symbol', 'timestamp', 'target', 'target_label']
        available_features = [col for col in features_df.columns if col not in exclude_cols]
        
        # Ensure we have the required features
        missing_features = set(self.feature_columns) - set(available_features)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Fill missing features with zeros
            for feature in missing_features:
                features_df = features_df.with_columns(pl.lit(0.0).alias(feature))
        
        # Extract features in correct order
        X = features_df.select(self.feature_columns).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        signals = []
        
        for i, row in enumerate(features_df.iter_rows(named=True)):
            symbol = row['symbol']
            timestamp = row['timestamp']
            
            # Get features for this row
            features_vector = X[i:i+1]
            
            # Generate predictions from all models
            model_predictions = {}
            model_probabilities = {}
            
            for model_name, model in self.trained_models.items():
                try:
                    # Scale features if needed
                    if model_name == 'logistic':
                        features_scaled = self.scaler.transform(features_vector)
                        prediction = model.predict(features_scaled)[0]
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(features_scaled)[0]
                    else:
                        prediction = model.predict(features_vector)[0]
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(features_vector)[0]
                    
                    model_predictions[model_name] = prediction
                    
                    if hasattr(model, 'predict_proba'):
                        # Convert to probability dictionary
                        prob_dict = {
                            'SELL': probabilities[0],
                            'HOLD': probabilities[1], 
                            'BUY': probabilities[2]
                        }
                        model_probabilities[model_name] = prob_dict
                        
                except Exception as e:
                    logger.warning(f"Failed to predict with {model_name} for {symbol}: {e}")
                    continue
            
            if not model_predictions:
                logger.warning(f"No predictions generated for {symbol}")
                continue
            
            # Ensemble prediction (majority vote)
            prediction_counts = {0: 0, 1: 0, 2: 0}  # SELL, HOLD, BUY
            for pred in model_predictions.values():
                prediction_counts[pred] += 1
            
            # Get final prediction
            final_prediction = max(prediction_counts, key=prediction_counts.get)
            signal_labels = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            final_signal = signal_labels[final_prediction]
            
            # Calculate confidence (agreement between models)
            total_models = len(model_predictions)
            max_agreement = max(prediction_counts.values())
            confidence = max_agreement / total_models if total_models > 0 else 0.0
            
            # Calculate signal strength (based on probabilities and features)
            signal_strength = confidence * 10.0
            
            # Add feature influence
            if row.get('unusual_activity_score', 0) > 5:
                signal_strength += 1.0
            
            if abs(row.get('net_gamma_exposure', 0)) > 1000:
                signal_strength += 0.5
            
            if abs(row.get('price_change_pct', 0)) > 2:
                signal_strength += 0.5
            
            signal_strength = min(signal_strength, 10.0)
            
            # Generate reasoning
            reasoning_parts = []
            if row.get('call_put_ratio', 1) > 1.5:
                reasoning_parts.append("High call activity")
            elif row.get('call_put_ratio', 1) < 0.67:
                reasoning_parts.append("High put activity")
            
            if row.get('net_gamma_exposure', 0) > 0:
                reasoning_parts.append("Positive gamma exposure")
            elif row.get('net_gamma_exposure', 0) < 0:
                reasoning_parts.append("Negative gamma exposure")
            
            if row.get('volume_ratio', 1) > 2:
                reasoning_parts.append("High volume")
            
            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Normal market conditions"
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                timestamp=timestamp,
                signal=final_signal,
                confidence=confidence,
                signal_strength=signal_strength,
                features_used={col: row.get(col, 0) for col in self.feature_columns[:5]},  # Top 5 features
                model_probabilities=model_probabilities,
                reasoning=reasoning
            )
            
            signals.append(signal)
        
        logger.info(f"Generated {len(signals)} signals")
        return signals
    
    def save_signals(self, signals: List[TradingSignal], output_dir: str = "data/signals"):
        """Save trading signals to file"""
        if not signals:
            logger.warning("No signals to save")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert signals to DataFrame
        signal_data = []
        for signal in signals:
            signal_data.append({
                'symbol': signal.symbol,
                'timestamp': signal.timestamp,
                'signal': signal.signal,
                'confidence': signal.confidence,
                'signal_strength': signal.signal_strength,
                'reasoning': signal.reasoning,
                'call_put_ratio': signal.features_used.get('call_put_ratio', 0),
                'net_gamma_exposure': signal.features_used.get('net_gamma_exposure', 0),
                'price_change_pct': signal.features_used.get('price_change_pct', 0),
                'volume_ratio': signal.features_used.get('volume_ratio', 1),
                'unusual_activity_score': signal.features_used.get('unusual_activity_score', 0)
            })
        
        signals_df = pl.DataFrame(signal_data)
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = Path(output_dir) / f"trading_signals_{timestamp}.parquet"
        
        signals_df.write_parquet(output_file)
        logger.info(f"Saved {len(signals)} signals to: {output_file}")
        
        # Also save as latest
        latest_file = Path(output_dir) / "trading_signals_latest.parquet"
        signals_df.write_parquet(latest_file)
        
        return output_file
    
    def get_top_signals(self, signals: List[TradingSignal], min_confidence: float = 0.6, 
                       top_n: int = 10) -> List[TradingSignal]:
        """Filter and return top trading signals"""
        # Filter by confidence
        filtered_signals = [s for s in signals if s.confidence >= min_confidence]
        
        # Sort by signal strength
        sorted_signals = sorted(filtered_signals, key=lambda x: x.signal_strength, reverse=True)
        
        return sorted_signals[:top_n]


def main():
    """Test the ML signal generator"""
    logger.info("🤖 Testing ML Signal Generator")
    
    # Initialize
    signal_generator = MLSignalGenerator()
    
    # Try to load existing models
    signal_generator.load_models()
    
    # If no models, train them
    if not signal_generator.trained_models:
        logger.info("No trained models found, training new models...")
        
        # Load training data
        training_data = signal_generator.load_training_data(lookback_days=7)
        
        if not training_data.is_empty():
            # Train models
            performances = signal_generator.train_models(training_data)
            
            # Show performance
            logger.info("\n📊 Model Performance:")
            for model_name, perf in performances.items():
                logger.info(f"  {model_name}: Accuracy={perf.accuracy:.3f}, CV={perf.cross_val_score:.3f}")
        else:
            logger.error("No training data available")
            return
    
    # Generate signals for test data
    logger.info("\n🎯 Generating test signals...")
    
    # Create some test features (in practice, these would come from your data pipeline)
    test_features = pl.DataFrame([
        {
            'symbol': 'AAPL',
            'timestamp': datetime.now(),
            'current_price': 150.0,
            'price_change': 2.5,
            'price_change_pct': 1.67,
            'volume_ratio': 1.5,
            'call_volume': 1000,
            'put_volume': 600,
            'call_put_ratio': 1.67,
            'net_gamma_exposure': 500,
            'dealer_gamma_exposure': -500,
            'flow_sentiment_bullish': 1.0,
            'flow_sentiment_bearish': 0.0,
            'unusual_activity_score': 3.5,
            'options_data_available': 1.0
        },
        {
            'symbol': 'MSFT',
            'timestamp': datetime.now(),
            'current_price': 300.0,
            'price_change': -1.5,
            'price_change_pct': -0.5,
            'volume_ratio': 0.8,
            'call_volume': 400,
            'put_volume': 800,
            'call_put_ratio': 0.5,
            'net_gamma_exposure': -300,
            'dealer_gamma_exposure': 300,
            'flow_sentiment_bullish': 0.0,
            'flow_sentiment_bearish': 1.0,
            'unusual_activity_score': 2.0,
            'options_data_available': 1.0
        }
    ])
    
    # Generate signals
    signals = signal_generator.generate_signal(test_features)
    
    # Show results
    if signals:
        logger.info(f"\n📈 Generated {len(signals)} signals:")
        for signal in signals:
            logger.info(f"  {signal.symbol}: {signal.signal} "
                       f"(confidence: {signal.confidence:.2f}, "
                       f"strength: {signal.signal_strength:.1f})")
            logger.info(f"    Reasoning: {signal.reasoning}")
        
        # Save signals
        signal_generator.save_signals(signals)
        
        # Get top signals
        top_signals = signal_generator.get_top_signals(signals, min_confidence=0.5)
        logger.info(f"\n🏆 Top {len(top_signals)} signals:")
        for signal in top_signals:
            logger.info(f"  {signal.symbol}: {signal.signal} "
                       f"(strength: {signal.signal_strength:.1f})")
    else:
        logger.warning("No signals generated")


if __name__ == "__main__":
    main()