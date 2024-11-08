import os
import datetime
import numpy as np
from fetch_data import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from classifiers import ML_DecisionTree, ML_KNN, ML_SVM
from gridsearchML import GS_DecisionTree

def createOutputDir():
    """ Create output directory with a timestamped subfolder """
    
    output_dir = 'output'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_subdir = os.path.join(output_dir, timestamp)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir, exist_ok=True)
    return output_subdir

def saveFeatureCorrelationHeatmap(trainX, feature_names, output_subdir):
    """ Save feature correlation heatmap """
    plt.figure(figsize=(16, 12))
    corr = np.corrcoef(trainX, rowvar=False)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)  # Red to green to red
    ax4 = sns.heatmap(corr, annot=True, xticklabels=feature_names, yticklabels=feature_names, cmap=cmap, center=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'correlation_heatmap.png'))
    plt.close()

def saveScatterPlots(trainX, trainY, feature_names, output_subdir, le):
    """ Save scatter plots of features in the dataset """
    scatter_plot_dir = os.path.join(output_subdir, 'Scatter plots')
    if not os.path.exists(scatter_plot_dir):
        os.makedirs(scatter_plot_dir, exist_ok=True)
    
    for i in range(0, trainX.shape[1]):
        for j in range(i + 1, trainX.shape[1]):
            fig1 = plt.figure()
            ax1 = sns.scatterplot(x=trainX[:, i], y=trainX[:, j], hue=le.inverse_transform(trainY))
            ax1.set_title(f"Scatter plot of {feature_names[i]} and {feature_names[j]}")
            ax1.set_xlabel(feature_names[i])
            ax1.set_ylabel(feature_names[j])
            plt.tight_layout()
            plt.savefig(os.path.join(scatter_plot_dir, f'scatter_plot_{feature_names[i]}_{feature_names[j]}.png'))
            plt.close(fig1)  # Close the figure after saving

def saveBoxPlots(trainX, trainY, feature_names, output_subdir, le):
    """ Save boxplots for all features """
    box_plot_dir = os.path.join(output_subdir, 'Feature distribution')
    if not os.path.exists(box_plot_dir):
        os.makedirs(box_plot_dir, exist_ok=True)
    
    for i in range(0, trainX.shape[1]):
        fig2 = plt.figure()
        ax2 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:,i])
        ax2.set_title(feature_names[i])
        ax2.set_ylabel(feature_names[i])
        plt.tight_layout()
        plt.savefig(os.path.join(box_plot_dir, 'boxplot_' + feature_names[i] + '.png'))
        plt.close(fig2)

def saveTargetDistribution(trainY, unique_targets, output_subdir):
    """ Save target distribution plot """
    plt.figure()
    ax = sns.countplot(x=trainY, color="skyblue")
    ax.set_xticks(np.arange(len(unique_targets)))  # Set correct number of ticks
    ax.set_xticklabels(unique_targets)  # Set the tick labels
    ax.set_title('Target Distribution in ' + output_subdir)
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'target_distribution.png'))
    plt.close()

def saveHistograms(trainX, feature_names, output_subdir):
    """ Save histograms for all features """
    n_features = trainX.shape[1]  # Total number of features (including Hu Moments)
    fig0, ax0 = plt.subplots(2, (n_features + 1) // 2, figsize=(15, 10))

    # Flatten axis array to simplify plotting in the loop
    ax0 = ax0.flatten()

    for i in range(n_features):
        sns.histplot(trainX[:, i], bins=10, ax=ax0[i], kde=False)
        ax0[i].set_xlabel(feature_names[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'histogram_all_features.png'))
    plt.close(fig0)  # Close the figure after saving

def saveFeatureUniqueness(trainX, feature_names, output_subdir):
    """ Calculate and save feature uniqueness plot """
    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(trainX, rowvar=False)
    
    # Calculate the total (average) correlation for each feature
    total_correlation = np.mean(np.abs(corr_matrix), axis=1)
    
    # Combine feature names with their total correlation for easy sorting
    unique_features = sorted(zip(feature_names, total_correlation), key=lambda x: x[1])
    
    # Extract sorted feature names and their correlations
    sorted_feature_names = [feature for feature, _ in unique_features]
    sorted_correlations = [correlation for _, correlation in unique_features]
    
    # Plot and save the feature uniqueness
    plt.figure(figsize=(10, 8))
    sns.barplot(x=sorted_correlations, y=sorted_feature_names, hue=sorted_feature_names, palette="RdYlGn_r", dodge=False, legend=False)
    plt.xlabel('Average Correlation')
    plt.ylabel('Feature')
    plt.title('Feature Uniqueness (Least Correlation to Most)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'feature_uniqueness.png'))
    plt.close()

if __name__ == "__main__":
    """feature exploration"""
    
    # Save start time
    start_time = datetime.datetime.now()
    
    # Create output directory with a timestamped subfolder
    output_subdir = createOutputDir()
    output_dir = 'output'

    # Retrieve the file list
    fileList = loadFiles()

    # fetch the data
    gestures = datasetBuilder(fileList, output_dir)

    # Encode the categorical labels
    le = LabelEncoder()
    coded_labels = le.fit_transform(gestures.target)
    
    #GS_DecisionTree(gestures, coded_labels, le.classes_)
    
    ML_DecisionTree(gestures, coded_labels, le.classes_, output_subdir)
    
    ML_KNN(gestures, coded_labels, le.classes_, output_subdir, 5)
    
    ML_SVM(gestures, coded_labels, le.classes_, output_subdir)
    
    # partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(gestures.data, coded_labels, test_size=0.25, stratify=gestures.target, random_state=42)

    # Data preparation (note that a pipeline  would help here)
    trainX = StandardScaler().fit_transform(trainX)

    # Save target distribution
    saveTargetDistribution(trainY, gestures.unique_targets, output_subdir)
    
    # Save histograms for all features
    saveHistograms(trainX, gestures.feature_names, output_subdir)

    # Save scatter plots for all features
    saveScatterPlots(trainX, trainY, gestures.feature_names, output_subdir, le)
    
    # Save boxplots for all features
    saveBoxPlots(trainX, trainY, gestures.feature_names, output_subdir, le)

    # Save feature correlation heatmap
    saveFeatureCorrelationHeatmap(trainX, gestures.feature_names, output_subdir)
    
    # Save feature uniqueness plot
    saveFeatureUniqueness(trainX, gestures.feature_names, output_subdir)

    # Close all figures
    plt.close('all')
    
    # Make sure all code is executed before closing
    plt.show(block=True)
    
    # Print a message to indicate the end of the exploration
    print("[INFO] Done exploring features")
    
    # Save end time
    end_time = datetime.datetime.now()
    
    # Calculate and print duration
    duration = end_time - start_time
    print(f"[INFO] Duration: {duration}")