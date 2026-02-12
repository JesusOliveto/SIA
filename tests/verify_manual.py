import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.kmeans_numpy import KMeansNumpy
from src.kmeans_loop import KMeansLoop

def test_kmeans():
    print("Generating dummy data...")
    X = np.random.rand(100, 5)
    k = 3
    
    print("Testing KMeansNumpy...")
    try:
        model_np = KMeansNumpy(n_clusters=k, random_state=42, verbose=True)
        model_np.fit(X)
        print(f"Numpy Inertia: {model_np.inertia_}")
        print("Numpy test PASSED.")
    except Exception as e:
        print(f"Numpy test FAILED: {e}")
        
    print("\nTesting KMeansLoop...")
    try:
        model_loop = KMeansLoop(n_clusters=k, random_state=42, verbose=True)
        model_loop.fit(X)
        print(f"Loop Inertia: {model_loop.inertia_}")
        print("Loop test PASSED.")
    except Exception as e:
        print(f"Loop test FAILED: {e}")

if __name__ == "__main__":
    test_kmeans()
