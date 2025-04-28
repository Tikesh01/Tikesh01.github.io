import qiskit
print(qiskit.__version__)

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile

def create_quantum_circuit(n):
    qc = QuantumCircuit(n, n)
    qc.h(range(n))  # Create superposition of all possible states
    return qc

def apply_oracle(qc, n, target_state):
    """
    Oracle that marks the target state we're searching for
    target_state: binary string representing the number we want to find
    """
    # Convert target state to list of positions where we need X gates
    for qubit, bit in enumerate(reversed(target_state)):
        if bit == '0':
            qc.x(qubit)
    
    # Multi-controlled Z gate implementation
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)  # Using mcx instead of mct
    qc.h(n-1)
    
    # Uncompute X gates
    for qubit, bit in enumerate(reversed(target_state)):
        if bit == '0':
            qc.x(qubit)
    return qc

def grover_diffusion(qc, n):
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)  # Using mcx instead of mct
    qc.h(n-1)
    qc.x(range(n))
    qc.h(range(n))
    return qc

def run_simulation(qc):
    backend = AerSimulator()
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc).result()
    counts = result.get_counts()
    return counts

def quantum_search(n, target_number):
    # Convert target number to binary string of length n
    target_state = format(target_number, f'0{n}b')
    
    # Initialize circuit
    qc = create_quantum_circuit(n)
    
    # One Grover iteration
    qc = apply_oracle(qc, n, target_state)
    qc = grover_diffusion(qc, n)
    
    # Measure all qubits
    qc.measure(range(n), range(n))
    
    # Run and get results
    counts = run_simulation(qc)
    return counts

# Example: Search for numbers 1, 2, and 3
print("\nSearching for number 1:")
counts1 = quantum_search(3, 1)  # 001 in binary
print(f"Results when searching for 1 (001): {counts1}")

print("\nSearching for number 2:")
counts2 = quantum_search(3, 2)  # 010 in binary
print(f"Results when searching for 2 (010): {counts2}")

print("\nSearching for number 3:")
counts3 = quantum_search(3, 3)  # 011 in binary
print(f"Results when searching for 3 (011): {counts3}")

print("\nThis demonstrates quantum search capability for finding specific numbers in a superposition of states")