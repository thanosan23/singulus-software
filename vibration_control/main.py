import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
import logging
import time
from collections import deque
from dataclasses import dataclass
import psutil
import threading
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    prediction_latency: float
    control_latency: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    temperature: float
    prediction_accuracy: float
    model_confidence: float
    vibration_levels: np.ndarray
    system_health: Dict[str, float]

class PerformanceMonitor:
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.current_metrics = None
        self._running = True
        self._monitor_thread = threading.Thread(target=self._background_monitoring)
        self._monitor_thread.daemon = True
        
        self.baseline_vibration = np.zeros(3)
        self.critical_thresholds = {
            'prediction_latency': 0.1,
            'control_latency': 0.05,
            'cpu_usage': 90.0,
            'memory_usage': 90.0,
            'gpu_usage': 95.0,
            'temperature': 80.0,
            'vibration_threshold': 2.0
        }
        self._monitor_thread.start()

    def start(self):
        if not self._monitor_thread.is_alive():
            self._running = True
            self._monitor_thread = threading.Thread(target=self._background_monitoring)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()

    def update_metrics(self, metrics: SystemMetrics):
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
        self._check_thresholds(metrics)
    
    def get_performance_summary(self) -> Dict[str, float]:
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-100:]
        return {
            'avg_prediction_latency': np.mean([m.prediction_latency for m in recent_metrics]),
            'avg_control_latency': np.mean([m.control_latency for m in recent_metrics]),
            'avg_prediction_accuracy': np.mean([m.prediction_accuracy for m in recent_metrics]),
            'avg_model_confidence': np.mean([m.model_confidence for m in recent_metrics]),
            'system_stability': self._calculate_stability_score(recent_metrics)
        }

    def _calculate_stability_score(self, metrics: List[SystemMetrics]) -> float:
        vibration_scores = [np.mean(m.vibration_levels) for m in metrics]
        return 1.0 - (np.std(vibration_scores) / (np.mean(vibration_scores) + 1e-8))

    def _check_thresholds(self, metrics: SystemMetrics):
        violations = []
        
        if metrics.prediction_latency > self.critical_thresholds['prediction_latency']:
            violations.append(f"High prediction latency: {metrics.prediction_latency:.3f}s")
        
        if metrics.control_latency > self.critical_thresholds['control_latency']:
            violations.append(f"High control latency: {metrics.control_latency:.3f}s")
        
        if metrics.cpu_usage > self.critical_thresholds['cpu_usage']:
            violations.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.gpu_usage > self.critical_thresholds['gpu_usage']:
            violations.append(f"High GPU usage: {metrics.gpu_usage:.1f}%")
        
        if violations:
            self._handle_violations(violations)

    def _handle_violations(self, violations: List[str]):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for violation in violations:
            logger.warning(f"[{timestamp}] Performance threshold violation: {violation}")

    def _background_monitoring(self):
        while self._running:
            try:
                if torch.cuda.is_available():
                    gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                    gpu_temp = 0.0
                else:
                    gpu_usage = 0.0
                    gpu_temp = 0.0

                system_metrics = SystemMetrics(
                    prediction_latency=0.0,
                    control_latency=0.0,
                    cpu_usage=psutil.cpu_percent(),
                    memory_usage=psutil.virtual_memory().percent,
                    gpu_usage=gpu_usage,
                    temperature=gpu_temp,
                    prediction_accuracy=0.0,
                    model_confidence=0.0,
                    vibration_levels=np.zeros(3),
                    system_health={'overall': 1.0}
                )
                
                self.update_metrics(system_metrics)
                time.sleep(1.0)
            
            except Exception as e:
                logger.error(f"Error in background monitoring: {str(e)}")

    def stop(self):
        self._running = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join()

class VibrationDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

class PrimaryNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(PrimaryNetwork, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.output_layer(features)

class VibrationController:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PrimaryNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.performance_monitor = PerformanceMonitor()
        self.prediction_history = deque(maxlen=1000)
        
        logger.info(f"Initialized VibrationController on device: {self.device}")

    def train(self, dataloader: DataLoader, epochs: int = 10):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self) -> np.ndarray:
        start_time = time.time()
        self.model.eval()
        
        input_data = torch.randn(1, 256).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_data)
            prediction = prediction.cpu().numpy()
        
        prediction_latency = time.time() - start_time
        
        metrics = SystemMetrics(
            prediction_latency=prediction_latency,
            control_latency=0.0,
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            gpu_usage=torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.is_available() else 0.0,
            temperature=0.0,
            prediction_accuracy=self._calculate_prediction_accuracy(),
            model_confidence=self._calculate_prediction_confidence(prediction),
            vibration_levels=self._get_current_vibration_levels(),
            system_health=self._assess_system_health()
        )
        
        self.performance_monitor.update_metrics(metrics)
        self.prediction_history.append(prediction)
        
        return prediction

    def _calculate_prediction_confidence(self, prediction: np.ndarray) -> float:
        if len(self.prediction_history) < 2:
            return 1.0
        previous_prediction = self.prediction_history[-1]
        stability = 1.0 - np.mean(np.abs(prediction - previous_prediction))
        return float(np.clip(stability, 0.0, 1.0))

    def _calculate_prediction_accuracy(self) -> float:
        if len(self.prediction_history) < 2:
            return 1.0
        recent_predictions = list(self.prediction_history)[-10:]
        accuracy = 1.0 - np.std([np.mean(pred) for pred in recent_predictions])
        return float(np.clip(accuracy, 0.0, 1.0))

    def _get_current_vibration_levels(self) -> np.ndarray:
        return np.array([
            np.random.normal(0.5, 0.1),
            np.random.normal(0.5, 0.1),
            np.random.normal(0.5, 0.1)
        ])

    def _assess_system_health(self) -> Dict[str, float]:
        return {
            'overall': 1.0,
            'sensor_health': 0.95,
            'model_health': 0.98,
            'control_health': 0.97
        }

def main():
    input_dim = 256
    hidden_dim = 128
    output_dim = 64
    controller = VibrationController(input_dim, hidden_dim, output_dim)
    
    num_samples = 1000
    example_data = np.random.randn(num_samples, input_dim)
    example_labels = np.random.randn(num_samples, output_dim)
    
    dataset = VibrationDataset(example_data, example_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    try:
        controller.train(dataloader)
        
        while True:
            prediction = controller.predict()
            performance_summary = controller.performance_monitor.get_performance_summary()
            logger.info(f"Performance Summary: {performance_summary}")
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        controller.performance_monitor.stop()
        logger.info("Shutting down monitoring system...")

if __name__ == "__main__":
    main()

