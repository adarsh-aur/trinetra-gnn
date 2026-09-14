import torch
from torch_geometric.loader import DataLoader
# The import path must be adapted based on where you run it.
try:
    from simple_gnn_demo import HybridCyberGNN # UPDATED CLASS NAME
    from demo_data_generator import DemoDataGenerator
except ImportError:
    try:
        from backend.demo_gnn.simple_gnn_demo import HybridCyberGNN
        from backend.demo_gnn.demo_data_generator import DemoDataGenerator
    except ImportError:
        # Fallback for complex environment setup
        from simple_gnn_demo import HybridCyberGNN
        from demo_data_generator import DemoDataGenerator

import os

# Train function now uses the fixed Hybrid model
def train_demo_model():
    """Train the Hybrid GNN model on synthetic data for demo"""
    MODEL_NAME = 'HYBRID' # Fixed name for saving
    print(f"🚀 Generating training data for {MODEL_NAME} model...")
    generator = DemoDataGenerator()
    graphs, labels = generator.generate_dataset(num_normal=200, num_attacks=200)
    
    # Add labels and placeholder for RGCN relation count
    print("   Adding labels to graphs...")
    num_relations = 1 # Required for RGCNConv initialization in the Hybrid model
    for i, graph in enumerate(graphs):
        graph.y = torch.tensor([labels[i]], dtype=torch.long)
        # Ensure all graphs have the relation count needed by the Hybrid model
        graph.num_relations = num_relations 
    
    # Split data
    print("   Splitting into train/test sets...")
    train_size = int(0.8 * len(graphs))
    train_graphs = graphs[:train_size]
    test_graphs = graphs[train_size:]
    
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # Initialize model
    print(f"\n🏗️ Initializing {MODEL_NAME} GNN model (GAT+GCN+SAGE+RGCN)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # Parameters matching the data generator features
    INPUT_DIM = 8
    HIDDEN_DIM = 32
    OUTPUT_DIM = 2

    # Initialize the Hybrid model
    model = HybridCyberGNN(
        input_dim=INPUT_DIM, 
        hidden_dim=HIDDEN_DIM, 
        output_dim=OUTPUT_DIM, 
        num_relations=num_relations
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()
    
    # Training loop
    print("\n🎯 Training model...\n")
    model.train()
    epochs = 50
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            total_loss += loss.item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")
    
    # Evaluate
    print("\n📊 Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    test_accuracy = correct / total
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    
    MODEL_FILENAME = f'models/demo_gnn_model_{MODEL_NAME}.pt' # Saving with HYBRID name
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': INPUT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'output_dim': OUTPUT_DIM,
        'model_type': MODEL_NAME, 
        'num_relations': num_relations,
        'test_accuracy': test_accuracy
    }, MODEL_FILENAME)
    
    print(f"💾 Model saved to {MODEL_FILENAME}")
    print(f"\n🎉 Training complete! {MODEL_NAME} Accuracy: {test_accuracy*100:.2f}%")
    return model

if __name__ == "__main__":
    train_demo_model()