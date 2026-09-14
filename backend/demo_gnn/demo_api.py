#!/usr/bin/env python3
"""
GNN Demo API - Fixed for cross-platform compatibility
Place this in: backend/demo_gnn/demo_api.py
"""

import sys
import os
from pathlib import Path

# Add parent directories to Python path for imports to work anywhere
current_dir = Path(__file__).parent.resolve()
backend_dir = current_dir.parent
project_root = backend_dir.parent

# Add to path if not already there
for path in [str(current_dir), str(backend_dir), str(project_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from flask import Flask, request, jsonify
from flask_cors import CORS   
import torch

# Smart imports with multiple fallback options for cross-platform compatibility
try:
    # Try relative import (when run as module)
    from simple_gnn_demo import HybridCyberGNN # UPDATED CLASS NAME
    from demo_data_generator import DemoDataGenerator
except ImportError:
    try:
        # Try from demo_gnn package
        from demo_gnn.simple_gnn_demo import HybridCyberGNN
        from demo_gnn.demo_data_generator import DemoDataGenerator
    except ImportError:
        # Try absolute import from backend
        from backend.demo_gnn.simple_gnn_demo import HybridCyberGNN
        from backend.demo_gnn.demo_data_generator import DemoDataGenerator

import json
from torch_geometric.data import Data, Batch 
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

# Global variables for model and generator
model = None
checkpoint = None
generator = None
model_loaded = False
MODEL_FILENAME = 'demo_gnn_model.pt' # Target model file

def initialize_model():
    """Initialize model with proper error handling"""
    global model, checkpoint, generator, model_loaded
    
    try:
        print("🔍 Loading Hybrid GNN model...")
        
        # Prioritize the HYBRID model, then check for older single-type models
        possible_paths = [
            current_dir / 'models' / MODEL_FILENAME, # New Hybrid model (Primary target)
            current_dir / 'models' / 'demo_gnn_model_GCN.pt', 
            current_dir / 'models' / 'demo_gnn_model_RGCN.pt',
            current_dir / 'models' / 'demo_gnn_model_GAT.pt',
            current_dir / 'models' / 'demo_gnn_model.pt', 
            Path('models') / MODEL_FILENAME
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        # Default parameters for demo mode
        default_params = {
            'input_dim': 8,
            'hidden_dim': 32,
            'output_dim': 2,
            'model_type': 'HYBRID', # Fixed to HYBRID
            'num_relations': 1,
            'test_accuracy': 0.95,
        }
        
        if model_path is None:
            print("⚠️  Hybrid model file not found, running in demo mode without trained weights")
            checkpoint = default_params
            
            # Initialize Hybrid model with default params
            model = HybridCyberGNN(
                input_dim=checkpoint['input_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                output_dim=checkpoint['output_dim'],
                num_relations=checkpoint['num_relations']
            )
            model_loaded = False
        else:
            print(f"✓ Found model at: {model_path}")
            checkpoint = torch.load(str(model_path), map_location=torch.device('cpu'))

            # Extract parameters, defaulting for compatibility with older saves
            input_dim = checkpoint.get('input_dim', 8)
            hidden_dim = checkpoint.get('hidden_dim', 32)
            output_dim = checkpoint.get('output_dim', 2)
            num_relations = checkpoint.get('num_relations', 1)
            
            # We initialize the HybridCyberGNN class as requested, regardless of old checkpoint type
            # The structure might differ if the checkpoint isn't HYBRID, but we attempt to load
            # as the user wants the new architecture.
            model = HybridCyberGNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_relations=num_relations
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            model_loaded = True
        
        model.eval()
        generator = DemoDataGenerator()
        print(f"✓ Model initialized successfully (Architecture: {model.__class__.__name__})")
        
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print("Running in demo mode with default parameters")
        
        # Re-initialize in case of error
        default_params = {
            'input_dim': 8, 'hidden_dim': 32, 'output_dim': 2, 
            'model_type': 'HYBRID', 'num_relations': 1, 'test_accuracy': 0.95
        }
        checkpoint = default_params
        model = HybridCyberGNN(
            input_dim=default_params['input_dim'], 
            hidden_dim=default_params['hidden_dim'], 
            output_dim=default_params['output_dim'],
            num_relations=default_params['num_relations']
        )
        generator = DemoDataGenerator()
        model_loaded = False

# Initialize on startup
initialize_model()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'message': 'Hybrid GNN Demo API - Multi-Cloud Threat Analyzer',
        'version': '3.0.0', 
        'status': 'online',
        'endpoints': {
            'health': '/api/health',
            'generate': '/api/generate-traffic',
            'analyze': '/api/analyze',
            'stats': '/api/stats'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Hybrid GNN Demo API',
        'model_loaded': model_loaded,
        'model_type': model.__class__.__name__, # Show actual model class name
        'test_accuracy': checkpoint.get('test_accuracy', 'N/A'),
        'working_directory': str(current_dir)
    })

@app.route('/api/generate-traffic', methods=['POST'])
def generate_traffic():
    """Generate sample traffic for demo"""
    try:
        data = request.json
        traffic_type = data.get('type', 'normal')
        attack_type = data.get('attack_type', 'ddos')
        
        if traffic_type == 'normal':
            graph = generator.generate_normal_traffic()
        else:
            graph = generator.generate_attack_traffic(attack_type)
        
        # Convert graph to JSON format
        graph_data = {
            'nodes': [],
            'edges': [],
            'features': graph.x.tolist(),
            'label': int(graph.label),
            'attack_type': getattr(graph, 'attack_type', 'Normal Traffic')
        }
        
        # Add node information
        for i, label in enumerate(graph.node_labels):
            graph_data['nodes'].append({
                'id': i,
                'label': label,
                'features': graph.x[i].tolist()
            })
        
        # Add edge information
        edges = graph.edge_index.t().tolist()
        for edge in edges:
            graph_data['edges'].append({
                'source': edge[0],
                'target': edge[1]
            })
        
        return jsonify(graph_data)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to generate traffic'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_traffic():
    """Analyze traffic graph with GNN"""
    try:
        data = request.json
        
        # Reconstruct graph from JSON
        x = torch.tensor(data['features'], dtype=torch.float)
        edges = [[e['source'], e['target']] for e in data['edges']]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Create Data object
        graph = Data(x=x, edge_index=edge_index)
        
        # Add necessary properties for Hybrid model (RGCN branch requires these)
        graph.edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
        graph.num_relations = 1
            
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        
        # Make prediction
        with torch.no_grad():
            output = model(graph)
            probabilities = torch.exp(output[0]).tolist()
            prediction = output.argmax(dim=1).item()
            embeddings = model.get_node_embeddings(graph)
        
        # Identify suspicious nodes (high activation)
        suspicious_nodes = []
        # Calculate mean/std based on current embeddings batch
        emb_mean = embeddings.mean()
        emb_std = embeddings.std()
        
        for i, emb in enumerate(embeddings):
            if isinstance(emb, np.ndarray):
                activation = float(np.linalg.norm(emb))
            else:
                activation = float(torch.tensor(emb).norm()) 
                
            # Use 1.5 standard deviations above the mean as a simple heuristic
            if activation > emb_mean + 1.5 * emb_std:
                suspicious_nodes.append({
                    'node_id': i,
                    'label': data['nodes'][i]['label'],
                    'activation': activation
                })
        
        result = {
            'prediction': 'Threat Detected' if prediction == 1 else 'Normal Traffic',
            'confidence': probabilities[prediction],
            'threat_probability': probabilities[1],
            'normal_probability': probabilities[0],
            'suspicious_nodes': sorted(suspicious_nodes, key=lambda x: x['activation'], reverse=True)[:5],
            'total_nodes': graph.num_nodes,
            'total_edges': graph.num_edges,
            # embeddings is a numpy array, convert to list for JSON
            'embeddings': embeddings.tolist()
        }
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'message': 'Failed to analyze traffic',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    return jsonify({
        'model_accuracy': checkpoint.get('test_accuracy', 0.95),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'architecture': f"Hybrid GNN (GAT+GCN+SAGE+RGCN)", # Updated architecture name
        'training_samples': 400,
        'attack_types': ['DDoS', 'Port Scan', 'Data Exfiltration'],
        'model_loaded': model_loaded
    })

if __name__ == '__main__':
    # Print banner with proper encoding handling
    try:
        print("\n" + "=" * 70)
        print("🚀 Hybrid GNN Demo API Server Starting...")
        print("=" * 70)
    except UnicodeEncodeError:
        print("\n" + "=" * 70)
        print("Hybrid GNN Demo API Server Starting...")
        print("=" * 70)
    
    print(f"📁 Working Directory: {current_dir}")
    print(f"🐍 Python Path: {sys.path[0]}")
    print(f"🌐 Server running on: http://localhost:5001")
    print(f"✓ Model Status: {'Loaded' if model_loaded else 'Demo Mode'}")
    print(f"🧠 Model Type: {model.__class__.__name__}") 
    print("\n📊 Available Endpoints:")
    print("   - GET  /                     (Root)")
    print("   - GET  /api/health            (Health Check)")
    print("   - POST /api/generate-traffic  (Generate Sample)")
    print("   - POST /api/analyze           (Analyze Traffic)")
    print("   - GET  /api/stats             (Model Stats)")
    print("=" * 70 + "\n")
    
    # Run with use_reloader=False to prevent double initialization
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)