import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
import time

st=time.time()
# Function to create an oracle for Grover's algorithm. The oracle marks the target index by applying a phase flip.
# It uses different strategies based on the number of qubits to handle the phase kickback and multi-controlled operations.
def create_oracle(values, target_idx, num_qubits):
    oracle = QuantumCircuit(num_qubits)
    
    # Apply phase kickback for the target index
    for i in range(num_qubits):
        if (target_idx >> i) & 1:
            oracle.x(i)
    
    # Handle different cases based on number of qubits
    if num_qubits == 1:
        # For single qubit, just apply phase flip if needed
        oracle.h(0)
        oracle.z(0)  # Phase flip
        oracle.h(0)
    elif num_qubits > 3:
        # Use intermediate qubits for large controls
        mid = num_qubits // 2
        oracle.h(num_qubits - 1)
        oracle.mcx(list(range(mid)), mid)
        oracle.mcx(list(range(mid, num_qubits - 1)), num_qubits - 1)
        oracle.h(num_qubits - 1)
    else:
        # For 2-3 qubits, use direct mcx
        oracle.h(num_qubits - 1)
        if num_qubits == 2:
            oracle.cx(0, 1)
        else:
            oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    
    # Restore input states
    for i in range(num_qubits):
        if (target_idx >> i) & 1:
            oracle.x(i)
    return oracle

# Function to create the diffusion operator (also known as the Grover operator) for Grover's algorithm.
# The diffusion operator amplifies the amplitude of the marked state by inverting the amplitudes about the mean.
def create_diffusion(num_qubits):
    # Create circuit with one additional ancilla qubit
    diffusion = QuantumCircuit(num_qubits + 1)
    
    # Apply H gates to data qubits
    for qubit in range(num_qubits):
        diffusion.h(qubit)
    
    # Apply X gates to data qubits
    for qubit in range(num_qubits):
        diffusion.x(qubit)
    
    # Use the last qubit as target and apply controlled operations in chunks
    chunk_size = 3  # Maximum number of control qubits per operation
    for i in range(0, num_qubits - 1, chunk_size):
        control_qubits = list(range(i, min(i + chunk_size, num_qubits - 1)))
        if len(control_qubits) > 0:
            diffusion.h(num_qubits)  # Apply H to ancilla
            diffusion.mcx(control_qubits, num_qubits)  # Multi-controlled X with limited controls
            diffusion.h(num_qubits)  # Apply H to ancilla
    
    # Apply X gates to data qubits
    for qubit in range(num_qubits):
        diffusion.x(qubit)
    
    # Apply H gates to data qubits
    for qubit in range(num_qubits):
        diffusion.h(qubit)
    
    return diffusion

# Function to find the index of the minimum value in a list using Grover's algorithm.
# It initializes a quantum circuit, applies Grover's algorithm, and measures the result to find the minimum index.
def grover_find_min_index(values):
    n = len(values)
    num_bits = max(1, int(np.ceil(np.log2(n))))  # Ensure at least 1 qubit
    
    # Find the minimum value index first (this will be our marked state)
    min_idx = np.argmin(values)
    
    # Create quantum circuit with an extra ancilla qubit
    qr = QuantumRegister(num_bits + 1, 'q')  # Add one ancilla qubit
    cr = ClassicalRegister(num_bits, 'c')     # We only measure the data qubits
    circuit = QuantumCircuit(qr, cr)
    
    # Initialize superposition on data qubits
    for i in range(num_bits):
        circuit.h(qr[i])
    
    # Number of Grover iterations
    iterations = int(np.pi/4 * np.sqrt(2**num_bits))
    
    # Apply Grover's algorithm
    oracle = create_oracle(values, min_idx, num_bits + 1)
    diffusion = create_diffusion(num_bits)
    
    for _ in range(iterations):
        circuit = circuit.compose(oracle)
        circuit = circuit.compose(diffusion)
    
    # Measure data qubits
    for i in range(num_bits):
        circuit.measure(qr[i], cr[i])
    
    # Run the circuit
    backend = Aer.get_backend('aer_simulator')
    result = backend.run(circuit, shots=1000).result()
    counts = result.get_counts()
    
    # Get most frequent result
    max_count_result = max(counts.items(), key=lambda x: x[1])[0]
    measured_index = int(max_count_result, 2)
    
    return measured_index % n

# Function to sort a DataFrame cluster using Grover's algorithm to find the minimum value iteratively.
# It repeatedly finds the minimum value in the cluster and appends it to the sorted list until the cluster is empty.
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

# Main function to perform cluster-based quantum sorting on a CSV file.
# It reads the input CSV, performs clustering, sorts each cluster using quantum_sort_cluster, and saves the final sorted data.
def cluster_based_quantum_sort(input_csv, sort_column, n_clusters=4, output_csv='cluster_sorted.csv'):
    #input_csv = file which is going to be sorted, sort_column -> column in csv file acc to which it will be cluster, n_clusters=4, how many clusters do you want.
    df = pd.read_csv(input_csv)
    df = df.dropna()
    
    if sort_column not in df.columns:# if  collumn is not in csv file raise error
        print(f"Column '{sort_column}' not found.")
        return

    print("Original Data:\n", df)

    # Classical Clustering 
    clustering_data = df[[sort_column]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)#KMeans is being applied which will give cluster ids
    df['cluster'] = kmeans.fit_predict(clustering_data) #a new col 'cluster' is created which holds the id of each row
    all_sorted = []
    #start grouping the clusters acc to similar ids
    for cluster_id in range(n_clusters):
        cluster_df = df[df['cluster'] == cluster_id].drop(columns=['cluster'])#it is df of similar clustered ids 
        print(f"\nSorting Cluster {cluster_id} (size {len(cluster_df)}):")
        #now quantum sort is going to be applied in each of cluster df -> cluster_df
        sorted_cluster = quantum_sort_cluster(cluster_df, sort_column)
        all_sorted.append(sorted_cluster)

    # Combine clusters and final classical sort
    merged_df = pd.concat(all_sorted, ignore_index=True)
    final_sorted_df = merged_df.sort_values(by=sort_column).reset_index(drop=True)

    print("\nFinal Sorted Data:")
    print(final_sorted_df)

    final_sorted_df.to_csv(output_csv, index=False)
    print(f"\nSorted data saved to '{output_csv}'.")

    et = time.time()
    tt = et-st
    print(f"Total time = {tt}")
if __name__ == "__main__":
    cluster_based_quantum_sort( 'iris - all-numbers.csv', sort_column="3.5", n_clusters=2)