import matplotlib.pyplot as plt
import numpy as np

runs = ["Run 1", "Run 2", "Run 3"]

uncoalesced = [0.714, 0.723, 0.722]
coalesced = [0.382, 0.368, 0.368]
matmul = [0.905, 0.901, 0.905]

x = np.arange(3)
width = 0.35

# Graph 1: Transpose comparison
plt.figure(figsize=(9, 5))
plt.bar(x - width/2, uncoalesced, width, label="Uncoalesced")
plt.bar(x + width/2, coalesced, width, label="Coalesced")
plt.xticks(x, runs)
plt.ylabel("Execution Time (ms)")
plt.title("CUDA Matrix Transpose Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/transpose_comparison.png", dpi=300)
plt.close()

# Graph 2: Matrix multiplication
plt.figure(figsize=(9, 5))
plt.bar(runs, matmul)
plt.ylabel("Execution Time (ms)")
plt.title("CUDA Naive Matrix Multiplication")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/matmul_performance.png", dpi=300)
plt.close()

# Graph 3: Average transpose comparison
labels = ["Uncoalesced", "Coalesced"]
averages = [np.mean(uncoalesced), np.mean(coalesced)]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, averages)
plt.ylabel("Average Execution Time (ms)")
plt.title("Average Matrix Transpose Execution Time")

for bar, value in zip(bars, averages):
    plt.text(bar.get_x() + bar.get_width()/2, value,
             f"{value:.3f} ms", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("results/average_transpose_comparison.png", dpi=300)
plt.close()

print("SUCCESS: All 3 graphs created!")
