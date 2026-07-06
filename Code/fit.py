#----------------------
# AUTHORS:
# DAMIAN - 10012545
# BHUVAN - 10001026
#-----------------------

import time
import psutil
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from logging import getLogger
logger = getLogger(__name__)

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model     = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device    = device

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def get_memory_usage(self):
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)  # Convert to MB
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024 ** 2)  # Convert to MB

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        peak_memory = self.get_memory_usage()
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            peak_memory = max(peak_memory, self.get_memory_usage())
            
        return running_loss / total, (correct / total) * 100, peak_memory

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        peak_memory = self.get_memory_usage()
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                peak_memory = max(peak_memory, self.get_memory_usage())
                
            return running_loss / total, (correct / total) * 100, peak_memory

    def fit(self, train_loader, val_loader, epochs, checkpoint_path=None):
        logger.info(" Starting Training Routine...")
        logger.info("-" * 50)
        start_time = time.perf_counter()
        best_val_acc = float("-inf")
        best_epoch = 0
        peak_training_memory = self.get_memory_usage()
        
        for epoch in range(epochs):
            train_loss, train_acc, train_peak_memory = self.train_one_epoch(train_loader)
            val_loss, val_acc, val_peak_memory = self.evaluate(val_loader)
            peak_training_memory = max(peak_training_memory, self.get_memory_usage())

            if checkpoint_path is not None and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                self.save_checkpoint(checkpoint_path)
            
            logger.info(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                        f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                        f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        if checkpoint_path is not None and best_epoch > 0:
            logger.info(f"Best checkpoint saved: {checkpoint_path} (epoch={best_epoch}, val_acc={best_val_acc:.2f}%)")
        
        logger.info("-" * 50)
        logger.info("Training Complete!")

        total_runtime = time.perf_counter() - start_time

        return best_val_acc, best_epoch, total_runtime, peak_training_memory

    def test_eval(self, dataloader):
        self.model.eval()
        all_preds, all_targets = [], []
        correct, total = 0, 0
        start_time = time.perf_counter()
        peak_inference_memory = self.get_memory_usage()
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
    
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
    
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(labels.cpu().tolist())

                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                peak_inference_memory = max(peak_inference_memory, self.get_memory_usage())
    
        precision = precision_score(all_targets, all_preds, average="macro")
        recall    = recall_score(all_targets, all_preds, average="macro")
        macro_f1  = f1_score(all_targets, all_preds, average="macro")
        accuracy  = (correct / total) * 100
        inference_latency = (time.perf_counter() - start_time) / total  # Average time per sample

        return precision, recall, macro_f1, accuracy, inference_latency, peak_inference_memory