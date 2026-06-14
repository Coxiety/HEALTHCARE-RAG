import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    import subprocess
    import sys
    print("Installing matplotlib and seaborn for visualization...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"])
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

def plot_metrics():
    # Metrics data
    categories = ['DISEASE', 'FOOD', 'NUTRIENT', 'Overall']
    precision = [91.29, 98.37, 95.80, 94.93]
    recall = [92.84, 98.98, 97.70, 96.39]
    f1 = [92.06, 98.67, 96.74, 95.65]

    x = np.arange(len(categories))
    width = 0.25  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    # Color palette
    colors_precision = '#3b82f6'  # Blue
    colors_recall = '#10b981'     # Emerald
    colors_f1 = '#8b5cf6'         # Violet

    # Plotting bars
    rects1 = ax.bar(x - width, precision, width, label='Precision', color=colors_precision, edgecolor='none', alpha=0.9)
    rects2 = ax.bar(x, recall, width, label='Recall', color=colors_recall, edgecolor='none', alpha=0.9)
    rects3 = ax.bar(x + width, f1, width, label='F1-Score', color=colors_f1, edgecolor='none', alpha=0.9)

    # Styling labels and title
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('HEALTHCARE-RAG: Detailed NER Evaluation Metrics by Class', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    
    # Legend
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=11)

    # Set y limit
    ax.set_ylim(80, 102)

    # Add values on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()

    # Save to artifacts path
    artifacts_dir = r"C:\Users\dongb\.gemini\antigravity-ide\brain\0c191100-b2c7-4c88-bb38-7b0d5c77cfa5"
    os.makedirs(artifacts_dir, exist_ok=True)
    save_path = os.path.join(artifacts_dir, "ner_metrics.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"Chart saved successfully at: {save_path}")

if __name__ == "__main__":
    plot_metrics()
