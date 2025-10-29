import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    
    try:
        data = pd.read_csv("analysis/metrics.csv")
    except(FileNotFoundError):
        print("the metrics file could not be found at 'analysis/metrics.csv'.")
        return
    
    names = data['dataset'].unique()
    miss_rates = data["miss_rate"].unique()
    modalities = data["miss_modality"].unique()

    fig , a = plt.subplots(1,len(names), figsize=(20, 4))   # #rows # cols
    fig.suptitle("rmse_mean")
    
    for i, name in enumerate(names):
        dataset = data[data['dataset'] == name]

        for modality in modalities:
            modality_set = dataset[dataset['miss_modality'] == modality]
            
            a[i].plot(miss_rates, modality_set['rmse_mean'], label=modality)
                
        a[i].set_title(name)
        a[i].set_xlabel("miss_rate")
        a[i].set_ylabel("rmse_mean")
        a[i].set_ylim(0, 0.5)
        
        
    fig.legend(a[0].get_lines(), modalities, loc="center right", ncol=1)
    
    plt.savefig("RMSE_Means.pdf", format='pdf', dpi=1200)


    fig2, a2 = plt.subplots(1,1, figsize=(20,4))
    fig2.suptitle("Mean Success Rate per Dataset")
    success_rates = []
    labels = []

    for name in names:
        for modality in modalities:
            val = data[(data['miss_modality'] == modality) & (data['dataset'] == name)]["success_rate"].mean()
            success_rates.append(val)
            labels.append(f'{name}_{modality}')

    # Convert to grouped data structure
    group_values = {name: [] for name in names}
    for name in names:
        for modality in modalities:
            val = data[(data['miss_modality'] == modality) & (data['dataset'] == name)]["success_rate"].mean()
            group_values[name].append(val)

    # Create grouped bars
    x = np.arange(len(names))  # positions for each dataset
    width = 0.2                   # width of each bar

    for i, modality in enumerate(modalities):
        modality_values = [group_values[name][i] for name in names]
        rects = a2.bar(x + i * width, modality_values, width, label=modality)
        a2.bar_label(rects)

    a2.set_xlabel("Dataset")
    a2.set_ylabel("Success Rate")
    a2.set_xticks(x + width * (len(modalities)-1)/2)
    a2.set_xticklabels(names)

    fig2.legend()

    plt.savefig("Success_rates.pdf", format='pdf', dpi=1200)
        
main()