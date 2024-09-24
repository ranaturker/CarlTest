import pandas as pd
import glob
import matplotlib
import matplotlib.pyplot as plt

# Adjust font rendering for compatibility with PDF/PS
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Color list for each policy
colors = ['blue', 'red', 'green']

# Directory where your result files are located
folder_name = '/Users/ranaturker/Desktop/results-mass-10'

# Policies and corresponding file patterns and display names
policies = ['sac_with_context', 'sac_without_context', 'sac_with_full_context']
policies_name_to_display = ["With Changing Context", "Without Context", "With Full Context"]


def plot_learning_curves():
    plt.figure(figsize=(8, 6))
    for j, policy in enumerate(policies):
        # Adjust file pattern to match your naming convention
        file_pattern = f"{folder_name}/{policy}_seed_*_SAC_1.csv"
        csv_files = glob.glob(file_pattern)

        # Debugging: Print matched files for each policy
        print(f"Matched files for policy '{policy}': {csv_files}")

        if not csv_files:
            print(f"No files found for pattern: {file_pattern}")
            continue  # Skip to the next policy if no files are found

        # Read CSV files and create a list of dataframes
        dataframes = [pd.read_csv(file, sep=',').set_index('Step')['Value'] for file in csv_files]

        # Calculate mean and standard deviation over different seeds
        df_mean = pd.concat(dataframes, axis=1).mean(axis=1)
        df_std = pd.concat(dataframes, axis=1).std(axis=1)

        # Plot mean and fill between standard deviation
        plt.plot(df_mean.index, df_mean, label=f'{policies_name_to_display[j]}', color=colors[j])
        plt.fill_between(df_mean.index, df_mean - df_std, df_mean + df_std, color=colors[j], alpha=0.2)

    plt.xlim(0, 100200)  # Assuming steps go up to 20000 based on your data
    plt.ylim(-1750, 0)  # Adjust to match the negative reward values
    plt.yticks(range(-1750, 1, 250))
    # Add a legend for each policy
    plt.legend(fontsize=10, loc='lower right')

    # Customize the tick size for x and y axes
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    # Set grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    # Set x-axis and y-axis labels
    plt.xlabel('Steps', fontsize=16)
    plt.ylabel('Reward', fontsize=16)

    # Add title to the plot
    plt.title('Training Mass Context With 10', fontsize=18)

    # Display the plot
    plt.show()


# Call the function to plot learning curves
plot_learning_curves()
