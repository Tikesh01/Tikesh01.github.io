import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer

def create_oracle(values, target_idx, num_qubits):
    oracle = QuantumCircuit(num_qubits)
    # Apply phase kickback for the target index
    for i in range(num_qubits):
        if (target_idx >> i) & 1:
            oracle.x(i)
    oracle.h(num_qubits - 1)
    oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    oracle.h(num_qubits - 1)
    for i in range(num_qubits):
        if (target_idx >> i) & 1:
            oracle.x(i)
    return oracle

def create_diffusion(num_qubits):
    diffusion = QuantumCircuit(num_qubits)
    # Apply H gates to all qubits
    for qubit in range(num_qubits):
        diffusion.h(qubit)
    # Apply X gates to all qubits
    for qubit in range(num_qubits):
        diffusion.x(qubit)
    # Apply multi-controlled Z gate
    diffusion.h(num_qubits - 1)
    diffusion.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    diffusion.h(num_qubits - 1)
    # Apply X gates to all qubits
    for qubit in range(num_qubits):
        diffusion.x(qubit)
    # Apply H gates to all qubits
    for qubit in range(num_qubits):
        diffusion.h(qubit)
    return diffusion

def grover_find_min_index(values):
    n = len(values)
    num_bits = int(np.ceil(np.log2(n)))
    
    # Find the minimum value index
    min_idx = np.argmin(values)
    
    # Create quantum circuit
    qr = QuantumRegister(num_bits)
    cr = ClassicalRegister(num_bits)
    circuit = QuantumCircuit(qr, cr)
    
    # Initialize superposition
    circuit.h(range(num_bits))
    
    # Number of Grover iterations
    iterations = int(np.pi/4 * np.sqrt(2**num_bits))
    
    # Apply Grover's algorithm
    oracle = create_oracle(values, min_idx, num_bits)
    diffusion = create_diffusion(num_bits)
    
    for _ in range(iterations):
        circuit = circuit.compose(oracle)
        circuit = circuit.compose(diffusion)
    
    # Measure
    circuit.measure(qr, cr)
    
    # Run the circuit using AerSimulator
    backend = Aer.get_backend('aer_simulator')
    result = backend.run(circuit, shots=1000).result()
    counts = result.get_counts()
    
    # Get most frequent result
    max_count_result = max(counts.items(), key=lambda x: x[1])[0]
    measured_index = int(max_count_result, 2)
    
    return measured_index % n

def quantum_sort_cluster(cluster_df, sort_column):
    df = cluster_df.copy().reset_index(drop=True)
    sorted_rows = []

    while not df.empty:
        values = df[sort_column].tolist()
        min_idx = grover_find_min_index(values)
        
        # Ensure min_idx is within bounds
        if min_idx >= len(df):
            min_idx = len(df) - 1
            
        sorted_rows.append(df.iloc[min_idx])
        df = df.drop(df.index[min_idx]).reset_index(drop=True)

    return pd.DataFrame(sorted_rows)

def cluster_based_quantum_sort(input_csv, sort_column, n_clusters=20, output_csv='cluster_sorted.csv'):
    df = pd.read_csv(input_csv)
    df = df.dropna()
    
    if sort_column not in df.columns:
        print(f"Column '{sort_column}' not found.")
        return

    print("Original Data:\n", df)

    # Clustering
    clustering_data = df[[sort_column]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(clustering_data)

    all_sorted = []
  
    for cluster_id in range(n_clusters):
        cluster_df = df[df['cluster'] == cluster_id].drop(columns=['cluster'])
        print(f"\nSorting Cluster {cluster_id} (size {len(cluster_df)}):")
        sorted_cluster = quantum_sort_cluster(cluster_df, sort_column)
        all_sorted.append(sorted_cluster)

    # Combine clusters and final classical sort
    merged_df = pd.concat(all_sorted, ignore_index=True)
    final_sorted_df = merged_df.sort_values(by=sort_column).reset_index(drop=True)

    print("\nFinal Sorted Data:")
    print(final_sorted_df)

    final_sorted_df.to_csv(output_csv, index=False)
    print(f"\nSorted data saved to '{output_csv}'.")

if __name__ == "__main__":
    cluster_based_quantum_sort("business-financial-data-december-2024-quarter-csv.csv", sort_column="Data_value", n_clusters=2)