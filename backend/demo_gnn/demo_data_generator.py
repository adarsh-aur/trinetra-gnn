import numpy as np
import torch
from torch_geometric.data import Data
import random

class DemoDataGenerator:
    """Generate realistic-looking network traffic graphs for demo"""
    
    def __init__(self):
        self.normal_ips = self._generate_ip_list(50)
        self.malicious_ips = self._generate_ip_list(10)
    
    def _generate_ip_list(self, count):
        """Generate fake IPs for demo"""
        return [f"192.168.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}" 
                for _ in range(count)]
    
    def generate_normal_traffic(self):
        """Generate normal network traffic graph""" 
        num_nodes = random.randint(15, 25)
        
        node_features = []
        node_labels = []
        
        for i in range(num_nodes):
            port = random.choice([80, 443, 22, 3306, 5432])
            packet_count = random.randint(10, 100)
            total_bytes = packet_count * random.randint(500, 1500)
            duration = random.uniform(0.1, 5.0)
            protocol = random.choice([1, 0])
            is_internal = random.choice([1, 1, 1, 0])
            avg_packet_size = total_bytes / packet_count
            
            features = [
                port / 65535,
                packet_count / 1000,
                total_bytes / 100000,
                duration / 10,
                protocol,
                1 - protocol,
                is_internal,
                avg_packet_size / 1500
            ]
            
            node_features.append(features)
            node_labels.append(self.normal_ips[i % len(self.normal_ips)])
        
        edges = []
        for i in range(num_nodes):
            num_connections = random.randint(1, 3)
            for _ in range(num_connections):
                target = random.randint(0, num_nodes - 1)
                if target != i:
                    edges.append([i, target])
        
        if len(edges) == 0:
            edges.append([0, 1])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(node_features, dtype=torch.float)
        
        graph = Data(x=x, edge_index=edge_index)
        graph.label = 0
        graph.attack_type = 'Normal Traffic'  # Add to all graphs
        graph.node_labels = node_labels
        
        # Placeholder for RGCN
        # If we had multiple edge types, we would generate them here.
        graph.edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
        
        return graph
    
    def generate_attack_traffic(self, attack_type='ddos'):
        """Generate attack traffic graph"""
        if attack_type == 'ddos':
            return self._generate_ddos_attack()
        elif attack_type == 'port_scan':
            return self._generate_port_scan()
        elif attack_type == 'data_exfiltration':
            return self._generate_data_exfiltration()
        else:
            return self._generate_ddos_attack()
    
    def _generate_ddos_attack(self):
        """DDoS: Many sources → One target"""
        num_attackers = random.randint(20, 30)
        num_normal = random.randint(5, 10)
        total_nodes = num_attackers + num_normal + 1
        
        node_features = []
        node_labels = []
        
        # Victim
        node_features.append([
            80 / 65535, 500 / 1000, 50000 / 100000, 10 / 10,
            1, 0, 1, 1000 / 1500
        ])
        node_labels.append("192.168.1.100 [VICTIM]")
        
        # Attackers
        for i in range(num_attackers):
            node_features.append([
                random.randint(1024, 65535) / 65535,
                random.randint(100, 200) / 1000,
                random.randint(10000, 30000) / 100000,
                random.uniform(5, 10) / 10,
                1, 0, 0, 500 / 1500
            ])
            node_labels.append(f"{self.malicious_ips[i % len(self.malicious_ips)]} [ATTACKER]")
        
        # Normal nodes
        for i in range(num_attackers + 1, total_nodes):
            node_features.append([
                random.choice([80, 443]) / 65535,
                random.randint(10, 50) / 1000,
                random.randint(5000, 15000) / 100000,
                random.uniform(0.5, 3) / 10,
                1, 0, 1, 800 / 1500
            ])
            node_labels.append(self.normal_ips[i % len(self.normal_ips)])
        
        # Edges: All attackers → victim
        edges = [[i, 0] for i in range(1, num_attackers + 1)]
        
        # Some normal traffic
        for i in range(num_attackers + 1, total_nodes):
            target = random.randint(0, total_nodes - 1)
            if target != i:
                edges.append([i, target])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(node_features, dtype=torch.float)
        
        graph = Data(x=x, edge_index=edge_index)
        graph.label = 1
        graph.attack_type = 'DDoS'
        graph.node_labels = node_labels
        
        # Placeholder for RGCN
        graph.edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)

        return graph
    
    def _generate_port_scan(self):
        """Port Scan: One source → Many targets"""
        num_targets = random.randint(15, 25)
        total_nodes = num_targets + 1
        
        node_features = []
        node_labels = []
        
        # Scanner
        node_features.append([
            random.randint(1024, 65535) / 65535,
            200 / 1000, 5000 / 100000, 8 / 10,
            1, 0, 0, 25 / 1500
        ])
        node_labels.append(f"{self.malicious_ips[0]} [SCANNER]")
        
        # Targets
        for i in range(num_targets):
            node_features.append([
                random.randint(1, 65535) / 65535,
                random.randint(5, 20) / 1000,
                random.randint(1000, 5000) / 100000,
                random.uniform(0.1, 1) / 10,
                1, 0, 1, 200 / 1500
            ])
            node_labels.append(self.normal_ips[i % len(self.normal_ips)])
        
        edges = [[0, i] for i in range(1, total_nodes)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(node_features, dtype=torch.float)
        
        graph = Data(x=x, edge_index=edge_index)
        graph.label = 1
        graph.attack_type = 'Port Scan'
        graph.node_labels = node_labels

        # Placeholder for RGCN
        graph.edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
        
        return graph
    
    def _generate_data_exfiltration(self):
        """Data Exfiltration: Large data transfer"""
        num_nodes = random.randint(10, 15)
        
        node_features = []
        node_labels = []
        
        # Compromised server
        node_features.append([
            3306 / 65535, 300 / 1000, 80000 / 100000, 15 / 10,
            1, 0, 1, 2667 / 1500
        ])
        node_labels.append("192.168.1.50 [COMPROMISED]")
        
        # External attacker
        node_features.append([
            443 / 65535, 300 / 1000, 80000 / 100000, 15 / 10,
            1, 0, 0, 2667 / 1500
        ])
        node_labels.append(f"{self.malicious_ips[0]} [C2 SERVER]")
        
        # Normal nodes
        for i in range(num_nodes - 2):
            node_features.append([
                random.choice([80, 443, 22]) / 65535,
                random.randint(10, 50) / 1000,
                random.randint(5000, 15000) / 100000,
                random.uniform(0.5, 3) / 10,
                1, 0, 1, 800 / 1500
            ])
            node_labels.append(self.normal_ips[i % len(self.normal_ips)])
        
        edges = [[0, 1], [1, 0]]
        
        for i in range(2, num_nodes):
            target = random.randint(0, num_nodes - 1)
            if target != i:
                edges.append([i, target])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(node_features, dtype=torch.float)
        
        graph = Data(x=x, edge_index=edge_index)
        graph.label = 1
        graph.attack_type = 'Data Exfiltration'
        graph.node_labels = node_labels

        # Placeholder for RGCN
        graph.edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
        
        return graph
    
    def generate_dataset(self, num_normal=100, num_attacks=100):
        """Generate full training dataset"""
        graphs = []
        labels = []
        
        print(f"   Generating {num_normal} normal traffic samples...")
        for _ in range(num_normal):
            graph = self.generate_normal_traffic()
            graphs.append(graph)
            labels.append(0)
        
        print(f"   Generating {num_attacks} attack samples...")
        attack_types = ['ddos', 'port_scan', 'data_exfiltration']
        for i in range(num_attacks):
            attack_type = attack_types[i % len(attack_types)]
            graph = self.generate_attack_traffic(attack_type)
            graphs.append(graph)
            labels.append(1)
        
        return graphs, labels