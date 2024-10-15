import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

def main():
    for name in ("bf_train_results",):
        csv_file = "results/" + name + ".csv"

        results_df = pd.read_csv(csv_file, header=0)

        
        stats_df = pd.DataFrame()

        for output_type in ("train_loss", "val_loss", "val_acc"):
            mask = results_df.columns.str.contains(output_type + '_.')

            stats_df.loc[:, output_type + "_avg"] = results_df.loc[:, mask].mean(axis=1)
            stats_df.loc[:, output_type + "_min"] = results_df.loc[:, mask].min(axis=1)
            stats_df.loc[:, output_type + "_max"] = results_df.loc[:, mask].max(axis=1)
        
        fig, ax = plt.subplots()

        x = np.linspace(0, 100, stats_df.shape[0])

        ax.plot(x, stats_df.loc[:, "train_loss_avg"])
        ax.plot(x, stats_df.loc[:, "val_loss_avg"])
        ax.plot(x, stats_df.loc[:, "val_acc_avg"])

        ax.legend(("train loss", "validation loss", "validation accuracy"))

        plt.savefig("averages_" + name + ".png")
    


if __name__ == "__main__":
    main()