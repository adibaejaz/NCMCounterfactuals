import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def json_to_excel(base_dir, excel_file):
    # Define ground truth graphs
    three_node_graphs = ["three_indep", "iv", "iv_equiv", "backdoor", "frontdoor", "three_unconst"]
    
    # List to collect data
    data_records = []
    
    # Traverse directories
    for ground_truth_graph in three_node_graphs:
        for trial_index in range(10):
            exp_dir = os.path.join(base_dir, f"graph={ground_truth_graph}-n_samples=10000-dim=1-trial_index={trial_index}")
            if os.path.exists(exp_dir):
                for test_graph in three_node_graphs:
                    result_path = os.path.join(exp_dir, test_graph, "results.json")
                    if os.path.exists(result_path):
                        with open(result_path, 'r') as f:
                            result_data = json.load(f)
                        
                        # Append relevant data
                        data_records.append({
                            "ground_truth_graph": ground_truth_graph,
                            "trial_index": trial_index,
                            "test_graph": test_graph,
                            **result_data
                        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data_records)
    
    # Compute averages and standard errors
    summary_df = df.groupby(["ground_truth_graph", "test_graph"]).agg(
        mean_total_true_KL=("total_true_KL", "mean"),
        std_total_true_KL=("total_true_KL", "std"),
        mean_total_dat_KL=("total_dat_KL", "mean"),
        std_total_dat_KL=("total_dat_KL", "std")
    ).reset_index()
    
    # Write to Excel
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Raw Data', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"Data successfully written to {excel_file}")
    
    # Generate bar charts
    for ground_truth_graph in three_node_graphs:
        subset = summary_df[summary_df["ground_truth_graph"] == ground_truth_graph]
        x = np.arange(len(subset))
        width = 0.4

        fig, ax = plt.subplots()
        bars1 = ax.bar(x - width/2, subset["mean_total_true_KL"], width, yerr=subset["std_total_true_KL"], label='total_true_KL', capsize=5)
        bars2 = ax.bar(x + width/2, subset["mean_total_dat_KL"], width, yerr=subset["std_total_dat_KL"], label='total_dat_KL', capsize=5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(subset["test_graph"], rotation=45, ha="right")
        ax.set_ylabel("KL Divergence")
        ax.set_title(f"{ground_truth_graph} KL Divergence Comparison")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"results/{ground_truth_graph}_kl_divergence.png")
        plt.close()
        
    print("Bar charts saved as PNG files.")

# Example usage
json_to_excel('out', 'results/results.xlsx')