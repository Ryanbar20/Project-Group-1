import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def dense_analysis():
    
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
            
            print(miss_rates, modality_set['rmse_mean'])
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


def ERRW_analysis():
    try:
        data = pd.read_csv("analysis/metrics.csv")
    except(FileNotFoundError):
        print("the metrics file could not be found at 'analysis/metrics.csv'.")
        return
    
    names = data["dataset"].unique()
    modalities = data["miss_modality"].unique()
    experiments = data[['miss_modality', 'generator_sparsity', 'discriminator_sparsity']].drop_duplicates()
    
    fig , a = plt.subplots(1,len(names) * len(modalities), figsize=(20, 4))   # #rows # cols
    index = -1
    for name in names:
        for modality in modalities:
            dataset = data[(data['dataset'] == name) & (data['miss_modality'] == modality)][['dataset','miss_modality', 'generator_sparsity', 'discriminator_sparsity', 'rmse_mean']]
            index += 1
            for gen_sparsity in dataset['generator_sparsity'].unique():
                x_y = dataset[dataset['generator_sparsity'] == gen_sparsity][['discriminator_sparsity', 'rmse_mean']]
                
                a[index].plot(x_y['discriminator_sparsity'], x_y['rmse_mean'])
                a[index].set_title(f"{name}_{modality}")
                if index == 0:
                    a[index].set_ylabel("rmse_mean")
                a[index].set_ylim(0,data['rmse_mean'].max() + 0.05)
    fig.suptitle("Mean RMSE for each discriminator sparsity per generator sparsity")
    fig.supxlabel("discriminator sparsity")
    # fig.subplots_adjust(left=0.071, bottom=0.11, right=0.9, top=0.848, wspace=0.329, hspace=0.24) # spacing used for graphs in report
    fig.legend(a[0].get_lines(), dataset['generator_sparsity'].unique(), loc="center right", ncol=1,title="generator_sparsity")
    plt.savefig("ERRW_Sparsities.pdf", format="pdf", dpi=1200)


# dense_analysis()
ERRW_analysis()