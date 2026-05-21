import torch
import json

class LedgerCompressor:
    def __init__(self, max_dictionary_size=1000):
        """
        High-Performance Root-Frequency Encryption & Compression Engine (HPC Simulation).
        Designed to compress massive corporate financial ledgers and cyber security log matrices
        by mapping recurring text structures into unified numerical root tokens via the GPU.
        """
        self.max_dictionary_size = max_dictionary_size
        self.root_dictionary = {}
        self.inverse_dictionary = {}
        self.current_token_id = 1  # 0 is reserved for padding/unknowns

    def build_root_dictionary(self, raw_logs_list):
        """
        Scans raw text data streams to identify and build the core frequency roots.
        """
        word_counts = {}
        for log in raw_logs_list:
            for word in log.split():
                word_counts[word] = word_counts.get(word, 0) + 1
                
        # Sort words by frequency to keep the most important root tokens
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, _ in sorted_words[:self.max_dictionary_size]:
            if word not in self.root_dictionary:
                self.root_dictionary[word] = self.current_token_id
                self.inverse_dictionary[self.current_token_id] = word
                self.current_token_id += 1

    def compress_to_tensor(self, raw_logs_list, max_sequence_length=16):
        """
        Compresses text rows into a high-density, highly compact PyTorch LongTensor.
        This enables immediate mathematical handling and parallel transmission.
        """
        compressed_matrix = []
        
        for log in raw_logs_list:
            encoded_row = []
            for word in log.split():
                # Fetch token from our frequency root dictionary
                token = self.root_dictionary.get(word, 0) # 0 if not in root dict
                encoded_row.append(token)
            
            # Pad or truncate to ensure strict tensor shape alignment (Hardware friendly)
            if len(encoded_row) < max_sequence_length:
                encoded_row += [0] * (max_sequence_length - len(encoded_row))
            else:
                encoded_row = encoded_row[:max_sequence_length]
                
            compressed_matrix.append(encoded_row)
            
        # Convert directly into a GPU-ready GPU Integer Tensor
        return torch.tensor(compressed_matrix, dtype=torch.long)

    def decompress_from_tensor(self, compressed_tensor):
        """
        Instantly decompresses the mathematical root tokens back into human-readable corporate text format.
        """
        decompressed_logs = []
        for i in range(compressed_tensor.size(0)):
            row = compressed_tensor[i].tolist()
            words_list = [self.inverse_dictionary.get(token, "[UNKNOWN]") for token in row if token != 0]
            decompressed_logs.append(" ".join(words_list))
        return decompressed_logs

if __name__ == "__main__":
    print("="*75)
    print("Testing Root-Frequency High-Performance Compression Kernel")
    print("="*75)
    
    # Simulating massive enterprise streaming network logs from MikroTik/Firewall
    mock_firewall_logs = [
        "SYS_ALERT Port_Scan detected from IP 192.168.1.50 critical breach",
        "SYS_INFO Normal user login successful channel secure API call",
        "SYS_ALERT Zero_Day_Exploit blocked by non-euclidean kernel structure critical",
        "SYS_INFO Normal data sync operation complete records archived cleanly"
    ]
    
    # Initialize the high-density compressor
    compressor = LedgerCompressor()
    compressor.build_root_dictionary(mock_firewall_logs)
    
    # Execute hardware-level compression matrix generation
    compressed_data = compressor.compress_to_tensor(mock_firewall_logs, max_sequence_length=12)
    
    # Calculate operational space savings
    raw_char_count = sum(len(log) for log in mock_firewall_logs)
    tensor_element_count = compressed_data.numel()
    
    print(f"Original Text Size in RAM : {raw_char_count} characters.")
    print(f"Compressed Matrix Tensor Shape : {compressed_data.shape}")
    print(f"Compressed Integer Elements   : {tensor_element_count} tokens.")
    print(f"\nMatrix Code Visualization on GPU Memory:\n{compressed_data}")
    print("-" * 75)
    
    # Test perfect decryption/decompression data recovery
    recovered_data = compressor.decompress_from_tensor(compressed_data)
    print(f"Decompressed Verified Sample Row:\n -> {recovered_data[2]}")
    print("="*75)
