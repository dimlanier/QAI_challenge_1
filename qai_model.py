import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import UGate, RYYGate, RXXGate
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import Aer

def load_breastmnist_data(num_train=400, num_test=100):
    """Load and preprocess BreastMNIST data"""
    from medmnist import BreastMNIST
    from torchvision import transforms
    
    data_transform = transforms.Compose([
        transforms.Resize((8, 8)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    train_dataset = BreastMNIST(split="train", transform=data_transform, download=True)
    test_dataset = BreastMNIST(split="test", transform=data_transform, download=True)
    
    x_train, y_train = [], []
    for i in range(min(num_train, len(train_dataset))):
        img, label = train_dataset[i]
        x_train.append(img.numpy())
        y_train.append(label)
    
    x_test, y_test = [], []
    for i in range(min(num_test, len(test_dataset))):
        img, label = test_dataset[i]
        x_test.append(img.numpy())
        y_test.append(label)
    
    x_train = np.array(x_train).reshape(len(x_train), -1)
    y_train = np.array(y_train).flatten()
    x_test = np.array(x_test).reshape(len(x_test), -1)
    y_test = np.array(y_test).flatten()
    
    return x_train, y_train, x_test, y_test

def amplitude_embedding_block(features, num_qubits, pad_with=0.0, label="AmpEmbed"):
    """Create normalized amplitude embedding circuit block"""
    features = np.asarray(features, dtype=float).flatten()
    
    target_dim = 2 ** num_qubits
    if len(features) < target_dim:
        features = np.pad(features, (0, target_dim - len(features)), constant_values=pad_with)
    else:
        features = features[:target_dim]
    
    norm = np.linalg.norm(features)
    if np.isclose(norm, 0, atol=1e-10):
        features = np.zeros(target_dim)
        features[0] = 1.0
    else:
        features = features / norm
        features = features / np.linalg.norm(features)
    
    qc = QuantumCircuit(num_qubits, name=label)
    qc.initialize(features, range(num_qubits))
    return qc.to_instruction()

def create_qcnn_circuit_from_diagram(features, params, num_qubits=6):
    """
    Create QCNN circuit based on the provided diagram architecture
    The diagram shows: Amplitude encoding -> U gates -> RYY/RXX -> Pooling -> Dense layers
    """
    qc = QuantumCircuit(num_qubits)
    
    # 1. Amplitude Embedding
    embed_gate = amplitude_embedding_block(features, num_qubits)
    qc.append(embed_gate, range(num_qubits))
    qc.barrier()
    
    # Parse parameters for different layers
    # Based on the diagram, we have U gates with 3 parameters, RYY/RXX with 1 parameter each
    param_idx = 0
    
    # 2. First U Gate Layer (6 U gates for 6 qubits)
    u_params1 = []
    for i in range(num_qubits):
        if param_idx + 2 < len(params):
            u_gate = UGate(params[param_idx], params[param_idx+1], params[param_idx+2])
            qc.append(u_gate, [i])
            u_params1.append([params[param_idx], params[param_idx+1], params[param_idx+2]])
            param_idx += 3
    qc.barrier()
    
    # 3. RYY Gate Layer (entangling gates)
    ryy_params = []
    for i in range(0, num_qubits-1, 2):
        if param_idx < len(params):
            ryy_gate = RYYGate(params[param_idx])
            qc.append(ryy_gate, [i, i+1])
            ryy_params.append(params[param_idx])
            param_idx += 1
    qc.barrier()
    
    # 4. Second U Gate Layer
    u_params2 = []
    for i in range(num_qubits):
        if param_idx + 2 < len(params):
            u_gate = UGate(params[param_idx], params[param_idx+1], params[param_idx+2])
            qc.append(u_gate, [i])
            u_params2.append([params[param_idx], params[param_idx+1], params[param_idx+2]])
            param_idx += 3
    qc.barrier()
    
    # 5. Pooling Layer (based on diagram - reduce from 6 to 3 qubits)
    # This is a simplified pooling - in practice you'd use measurement and conditional operations
    # For simulation, we'll use controlled rotations as pooling
    pool_params = []
    for i in range(0, num_qubits-1, 2):
        if param_idx < len(params):
            # Controlled rotation as pooling mechanism
            qc.cry(params[param_idx], i, i+1)
            pool_params.append(params[param_idx])
            param_idx += 1
    qc.barrier()
    
    # After pooling, we continue with remaining qubits (0, 2, 4)
    remaining_qubits = [0, 2, 4]
    
    # 6. Third U Gate Layer on remaining qubits
    u_params3 = []
    for i in remaining_qubits:
        if param_idx + 2 < len(params):
            u_gate = UGate(params[param_idx], params[param_idx+1], params[param_idx+2])
            qc.append(u_gate, [i])
            u_params3.append([params[param_idx], params[param_idx+1], params[param_idx+2]])
            param_idx += 3
    qc.barrier()
    
    # 7. RXX Gate Layer on remaining qubits
    rxx_params = []
    for i in range(len(remaining_qubits)-1):
        if param_idx < len(params):
            rxx_gate = RXXGate(params[param_idx])
            qc.append(rxx_gate, [remaining_qubits[i], remaining_qubits[i+1]])
            rxx_params.append(params[param_idx])
            param_idx += 1
    qc.barrier()
    
    # 8. Final U Gate Layer (dense layer equivalent)
    u_params_final = []
    for i in remaining_qubits:
        if param_idx + 2 < len(params):
            u_gate = UGate(params[param_idx], params[param_idx+1], params[param_idx+2])
            qc.append(u_gate, [i])
            u_params_final.append([params[param_idx], params[param_idx+1], params[param_idx+2]])
            param_idx += 3
    
    return qc

class QuantumCircuitFunction(torch.autograd.Function):
    @staticmethod
    def _run_circuit(features, params):
        """Run quantum circuit and return expectation values"""
        expectation_values = []
        num_qubits = 6
        
        # Define observable (Z on first qubit)
        observable = Operator(np.kron(np.diag([1, -1]), np.identity(2**(num_qubits-1))))
        
        for feature_vec in features:
            try:
                # Convert to numpy
                feature_np = feature_vec.detach().numpy() if isinstance(feature_vec, torch.Tensor) else feature_vec
                feature_np = feature_np.astype(np.float64)
                
                # Convert weights to numpy
                params_np = params.detach().numpy() if isinstance(params, torch.Tensor) else params
                
                # Create and run circuit
                qc = create_qcnn_circuit_from_diagram(feature_np, params_np, num_qubits)
                statevector = Statevector(qc)
                exp_val = statevector.expectation_value(observable).real
                expectation_values.append(exp_val)
                
            except Exception as e:
                print(f"Circuit execution error: {e}")
                # Return a fallback value that will be improved during training
                expectation_values.append(0.0)
                
        return torch.tensor(expectation_values, dtype=torch.float32)

    @staticmethod
    def forward(ctx, features, params):
        ctx.save_for_backward(features, params)
        return QuantumCircuitFunction._run_circuit(features, params)

    @staticmethod
    def backward(ctx, grad_output):
        features, params = ctx.saved_tensors
        gradients = torch.zeros_like(params)
        shift = np.pi / 2
        
        # Parameter shift rule for gradient computation
        for i in range(len(params)):
            params_up = params.clone()
            params_down = params.clone()
            params_up[i] += shift
            params_down[i] -= shift
            
            exp_val_up = QuantumCircuitFunction._run_circuit(features, params_up)
            exp_val_down = QuantumCircuitFunction._run_circuit(features, params_down)
            
            gradient = 0.5 * (exp_val_up - exp_val_down)
            gradients[i] = (grad_output * gradient).sum()
            
        return None, gradients

class QuantumLayer(nn.Module):
    def __init__(self, num_params):
        super(QuantumLayer, self).__init__()
        # Initialize parameters with small random values for better convergence
        self.params = nn.Parameter(torch.rand(num_params) * 0.1 * np.pi)
        
    def forward(self, features):
        return QuantumCircuitFunction.apply(features, self.params)

class DiagramQCNN(nn.Module):
    def __init__(self, num_quantum_params, num_classes):
        super(DiagramQCNN, self).__init__()
        # Classical preprocessing to ensure correct feature dimension
        self.pre_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Quantum layer based on the diagram architecture
        self.quantum_layer = QuantumLayer(num_quantum_params)
        # Classical post-processing
        self.classical_layer = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.pre_net(x)
        x = self.quantum_layer(x)
        x = x.unsqueeze(1)
        x = self.classical_layer(x)
        return x

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_balanced_dataloader(x_train, y_train, batch_size=16):
    """Create a balanced dataloader using weighted sampling"""
    # Convert to PyTorch tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    
    # Compute class weights for sampling
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight).double()
    
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    return DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

def calculate_metrics(predictions, targets):
    """Calculate precision, recall, and F1-score"""
    tp = ((predictions == 1) & (targets == 1)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return precision, recall, f1, accuracy

# Configuration based on the diagram architecture
NUM_QUBITS = 6
# Parameter count based on the diagram:
# 6 U gates (3 params each) = 18
# 3 RYY gates (1 param each) = 3  
# 6 U gates (3 params each) = 18
# 3 Pooling gates (1 param each) = 3
# 3 U gates (3 params each) = 9
# 2 RXX gates (1 param each) = 2
# 3 U gates (3 params each) = 9
# Total = 18 + 3 + 18 + 3 + 9 + 2 + 9 = 62 parameters
NUM_QUANTUM_PARAMS = 62
NUM_CLASSES = 2
BATCH_SIZE = 8  # Reduced for quantum simulation
EPOCHS = 15
LEARNING_RATE = 0.001

def visualize_circuit():
    """Visualize the QCNN circuit architecture"""
    # Create a sample circuit with random parameters
    sample_features = np.random.randn(64)
    sample_params = np.random.randn(NUM_QUANTUM_PARAMS) * 0.1
    
    qc = create_qcnn_circuit_from_diagram(sample_features, sample_params)
    
    print("QCNN Circuit Architecture:")
    print("=" * 50)
    print(f"Total qubits: {NUM_QUBITS}")
    print(f"Total quantum parameters: {NUM_QUANTUM_PARAMS}")
    print(f"Circuit depth: {qc.depth()}")
    print(f"Circuit operations: {qc.count_ops()}")
    print("\nCircuit structure:")
    print(qc.draw(output='text'))
    
    return qc

def main():
    print("Loading BreastMNIST data...")
    x_train, y_train, x_test, y_test = load_breastmnist_data(num_train=200, num_test=50)
    
    # Analyze class distribution
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    print(f"Training set class distribution: Class {train_unique[0]}: {train_counts[0]}, Class {train_unique[1]}: {train_counts[1]}")
    print(f"Imbalance ratio: {max(train_counts)/min(train_counts):.2f}:1")
    
    # Visualize the circuit architecture
    print("\n" + "="*60)
    print("QCNN CIRCUIT ARCHITECTURE")
    print("="*60)
    visualize_circuit()
    
    # Compute class weights for loss function
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"\nClass weights for loss function: {class_weights}")
    
    # Convert test data
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.LongTensor(y_test)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create balanced training loader
    train_loader = create_balanced_dataloader(x_train, y_train, BATCH_SIZE)
    
    # Initialize model
    model = DiagramQCNN(
        num_quantum_params=NUM_QUANTUM_PARAMS,
        num_classes=NUM_CLASSES
    )
    
    print(f"\nModel Configuration:")
    print(f"  - Quantum parameters: {NUM_QUANTUM_PARAMS}")
    print(f"  - Qubits: {NUM_QUBITS}")
    print(f"  - Feature dimension: 64 (8×8 images)")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'precision': [], 'recall': [], 'f1': []
    }
    
    print("\nStarting QCNN training with diagram-based architecture...")
    print("=" * 60)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.numpy())
            all_targets.extend(labels.numpy())
            
            if batch_idx % 5 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_precision, train_recall, train_f1, _ = calculate_metrics(
            torch.tensor(all_preds), torch.tensor(all_targets)
        )
        
        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.numpy())
                val_targets.extend(labels.numpy())
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100 * correct / total
        val_precision, val_recall, val_f1, _ = calculate_metrics(
            torch.tensor(val_preds), torch.tensor(val_targets)
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['precision'].append(val_precision)
        history['recall'].append(val_recall)
        history['f1'].append(val_f1)
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'\nEpoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.2f}s | LR: {current_lr:.6f}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}')
        print('-' * 60)
    
    # Plot comprehensive results
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 3)
    plt.plot(history['precision'], label='Precision')
    plt.plot(history['recall'], label='Recall')
    plt.plot(history['f1'], label='F1-Score')
    plt.title('Metrics vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 4)
    unique, counts = np.unique(y_train, return_counts=True)
    plt.bar(unique, counts, color=['skyblue', 'lightcoral'])
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(unique)
    
    plt.tight_layout()
    plt.show()
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL QCNN RESULTS - DIAGRAM ARCHITECTURE")
    print("="*60)
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Final Precision: {history['precision'][-1]:.4f}")
    print(f"Final Recall: {history['recall'][-1]:.4f}")
    print(f"Final F1-Score: {history['f1'][-1]:.4f}")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test_tensor)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_accuracy = 100 * (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        
        test_precision, test_recall, test_f1, _ = calculate_metrics(test_predicted, y_test_tensor)
        
        print(f"\nTest Set Performance:")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")

if __name__ == "__main__":
    main()