import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(data, bins=10, title="Histogram", xlabel="Values", ylabel="Frequency", save_as="histogram.png"):
    """
    Function to plot a histogram and save it as an image file.

    Parameters:
        data (list or array): Data to be plotted (e.g., sequence lengths).
        bins (int): Number of bins for the histogram.
        title (str): Title of the histogram.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_as (str): File name to save the plot (e.g., "histogram.png").
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True)
    plt.savefig(save_as)
    plt.close()
    
def plot_boxplot(data, title="Box Plot", xlabel="Data", ylabel="Values", save_as="boxplot.png"):
    """
    Function to plot a box plot and save it as an image file.

    Parameters:
        data (list or array): Data to be plotted (e.g., sequence lengths).
        title (str): Title of the box plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_as (str): File name to save the plot (e.g., "boxplot.png").
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, color='lightgreen')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.savefig(save_as)
    plt.close()

def plot_horizontal_bar(data, labels, title="Horizontal Bar Plot", xlabel="Values", ylabel="Categories", save_as="bar_plot.png"):
    """
    General function to plot a horizontal bar plot.

    Parameters:
        data (list): List of values (frequencies, probabilities, etc.).
        labels (list): List of category labels corresponding to the data.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_as (str): File name to save the plot (e.g., "plot.png").
    """
    if len(data) != len(labels):
        raise ValueError("The length of data and labels must be the same.")

    plt.figure(figsize=(10, 6))
    plt.barh(labels, data, color='skyblue')
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()

    plt.savefig(save_as)
    plt.close()

