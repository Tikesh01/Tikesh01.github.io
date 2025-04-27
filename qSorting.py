# cluster_based_quantum_sort("../customers-1000.csv",sort_column="Country",n_clusters=10)
from qiskit_aer import Aer
from qiskit import QuantumCircuit
import pandas as pd
import numpy as np

q = QuantumCircuit(10)
df=pd.read_csv("customers-1000.csv")
j=0
for i in df.columns:
    q[j] = df[i] 
    j+=1

def findDuplicates():
    pass

