import torch
import math

def generate_enterprise_cyber_traffic(n_samples=200):
    """
    Simulates highly complex, non-linearly intertwined network traffic logs.
    Includes normal employee workflows and sophisticated, stealthy Zero-Day/APT attacks.
    
    Returns:
        inputs (Tensor): Shape (n_samples, 12) representing 12 advanced network feature metrics.
        targets (Tensor): Shape (n_samples, 4) representing 4 operational security classes:
                          [Normal, Data_Exfiltration, Zero_Day_Exploit, Port_Scan]
    """
    torch.manual_seed(555) # For strict scientific replicability
    data_list = []
    target_list = []
    
    # 1. Generate Normal Corporate Traffic (Class 0)
    for _ in range(int(n_samples * 0.6)): # 60% of traffic is normal
        # Features: [packet_size, frequency, port_weight, timing_delta, ..., entropy]
        # Normal traffic has standard distributions but high variance due to various apps
        base_traffic = torch.randn(12) * 0.5 + torch.tensor([1.0, 0.5, -0.2, 0.1, 0.0, 0.9, -0.4, 0.3, 0.1, -0.1, 0.5, 0.2])
        data_list.append(base_traffic)
        target_list.append(torch.tensor([1.0, 0.0, 0.0, 0.0])) # Normal
        
    # 2. Generate Stealthy Data Exfiltration (Class 1)
    # Mimics slow, low-volume transfers hidden inside standard protocol anomalies
    for _ in range(int(n_samples * 0.15)):
        exfil_traffic = torch.randn(12) * 0.3 + torch.tensor([1.2, 0.4, -0.1, 0.15, 0.1, 0.8, -0.3, 0.4, 0.0, -0.2, 0.6, 0.3])
        data_list.append(exfil_traffic)
        target_list.append(torch.tensor([0.0, 1.0, 0.0, 0.0])) # Exfiltration
        
    # 3. Generate Sophisticated Zero-Day Exploits (Class 2)
    # Highly non-linear; deeply embedded within the geometric boundary of normal profiles
    for _ in range(int(n_samples * 0.15)):
        # Notice how the mean configuration deeply overlaps with normal corporate traffic
        zero_day = torch.randn(12) * 0.4 + torch.tensor([1.05, 0.48, -0.18, 0.12, 0.02, 0.88, -0.38, 0.32, 0.08, -0.12, 0.52, 0.22])
        data_list.append(zero_day)
        target_list.append(torch.tensor([0.0, 0.0, 1.0, 0.0])) # Zero-Day Exploit
        
    # 4. Generate Aggressive Distributed Port Scanning (Class 3)
    for _ in range(int(n_samples * 0.10)):
        scan_traffic = torch.randn(12) * 0.7 + torch.tensor([-0.5, 2.0, 1.5, -1.0, 0.8, -0.5, 1.0, -0.8, 0.5, 0.9, -0.2, 1.0])
        data_list.append(scan_traffic)
        target_list.append(torch.tensor([0.0, 0.0, 0.0, 1.0])) # Port Scan

    # Stack into structured tensors and shuffle completely to simulate real-time packet ingestion
    inputs = torch.stack(data_list)
    targets = torch.stack(target_list)
    
    shuffled_indices = torch.randperm(inputs.size(0))
    return inputs[shuffled_indices], targets[shuffled_indices]

if __name__ == "__main__":
    print("="*70)
    print("Testing Autonomous Cyber Traffic Generator Analytics")
    print("="*70)
    features, labels = generate_enterprise_cyber_traffic(n_samples=10)
    print(f"Generated Tensor Pipeline Shape : {features.shape}")
    print(f"Target Classification Matrix Shape: {labels.shape}")
    print(f"Sample Ingested Packet Feature Vector:\n{features[0].numpy()}")
    print("="*70)
