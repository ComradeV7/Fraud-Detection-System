import torch
import numpy as np
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

from src.model.anfis import ANFIS

def create_dataloader(
    X: np.ndarray, 
    y: np.ndarray, 
    batch_size: int = 256, 
    shuffle: bool = True
) -> DataLoader:
    """Converts numpy arrays into a PyTorch DataLoader for batch processing."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader

def train_epoch(
    model: ANFIS, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: torch.nn.Module,
    pos_weight: float = 1.0  # Added parameter for Class Weighting
) -> float:
    """Performs one full training pass over the dataset with class weighting."""
    model.train()
    total_loss = 0.0
    device = next(model.parameters()).device
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        
        # Calculate unreduced loss
        loss = criterion(predictions, y_batch)
        
        # Apply class weighting: Multiply the loss by pos_weight where the target is 1 (Fraud)
        weights = torch.ones_like(y_batch)
        weights[y_batch == 1] = pos_weight
        weighted_loss = (loss * weights).mean()
        
        weighted_loss.backward()
        optimizer.step()
        
        total_loss += weighted_loss.item()
        
    return total_loss / len(loader)

def evaluate(model: ANFIS, loader: DataLoader) -> dict:
    """Evaluates the model on unseen data and calculates classification metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    preds_binary = (preds >= 0.5).astype(int)
    
    metrics = {
        "auc": float(roc_auc_score(targets, preds)),
        "precision": float(precision_score(targets, preds_binary, zero_division=0)),
        "recall": float(recall_score(targets, preds_binary, zero_division=0)),
        "f1": float(f1_score(targets, preds_binary, zero_division=0))
    }
    
    return metrics

def train(
    model: ANFIS,
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray, 
    y_test: np.ndarray,
    epochs: int = 50,
    lr: float = 0.001,
    pos_weight: float = 1.0  # Added parameter for Class Weighting
) -> tuple[str, float, dict]:
    """Executes the training loop, tracks history, and registers in MLflow."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training utilizing hardware: {device} | Using Fraud Weight: {pos_weight:.2f}")
    
    model = model.to(device)
    train_loader = create_dataloader(X_train, y_train, batch_size=256, shuffle=True)
    test_loader = create_dataloader(X_test, y_test, batch_size=256, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Crucial Change: Set reduction to 'none' so we can apply custom batch weighting
    criterion = torch.nn.BCELoss(reduction='none') 
    
    # Initialize history tracking
    history = {'loss': [], 'auc': []}
    best_auc = 0.0
    best_model_state = None
    
    run = mlflow.start_run(run_name='anfis_training')
    run_id = run.info.run_id
    
    try:
        mlflow.log_params({"epochs": epochs, "learning_rate": lr, "batch_size": 256, "pos_weight": pos_weight})
        
        for epoch in range(1, epochs + 1):
            avg_loss = train_epoch(model, train_loader, optimizer, criterion, pos_weight)
            metrics = evaluate(model, test_loader)
            
            history['loss'].append(avg_loss)
            history['auc'].append(metrics['auc'])
            
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metrics(metrics, step=epoch)
            
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d}/{epochs} | Loss: {avg_loss:.4f} | Val AUC: {metrics['auc']:.4f}")
        
        model.load_state_dict(best_model_state)
        model = model.to("cpu")
        
        sample_input = torch.tensor(X_test[:5], dtype=torch.float32)
        sample_output = model(sample_input).detach().numpy()
        signature = infer_signature(X_test[:5], sample_output)
        
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="BAF_Fraud_ANFIS"
        )
        print(f"Training complete. Model registered in MLflow with AUC: {best_auc:.4f}")
        
    finally:
        mlflow.end_run()
        
    return run_id, best_auc, history