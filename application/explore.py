import os
import datetime
import numpy as np
from fetch_data import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from classefierSVM import ML_DecisionTree, ML_KNN, ML_SVM

if __name__ == "__main__":
    """feature exploration"""
    
    # Save start time
    start_time = datetime.datetime.now()
    
    # Create output directory with a timestamped subfolder
    output_dir = 'output'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_subdir = os.path.join(output_dir, timestamp)    
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir, exist_ok=True)

    #Features
    #area = 0
    #contour = 1
    #convexHullLength = 2
    #convexityDefects = 3
    #compactness = 4

    # get the data path
    fileList = loadFiles()

    # fetch the data
    gestures = datasetBuilder(fileList)

    # encode the categorical labels
    le = LabelEncoder()
    coded_labels = le.fit_transform(gestures.target)
    
    ML_DecisionTree(gestures, coded_labels, le.classes_)
    
    ML_KNN(gestures, coded_labels, le.classes_,5)
    
    ML_SVM(gestures, coded_labels, le.classes_)
    

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(gestures.data, coded_labels,
    test_size=0.25, stratify=gestures.target)#, random_state=42)

    # Data preparation (note that a pipeline  would help here)
    trainX = StandardScaler().fit_transform(trainX)

    # show target distribution
    ax = sns.countplot(x=trainY, color="skyblue")
    ax.set_xticks(np.arange(len(gestures.unique_targets)))  # Set correct number of ticks
    ax.set_xticklabels(gestures.unique_targets)  # Set the tick labels
    ax.set_title('Target Distribution in ' + output_subdir)
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'target_distribution.png'))
    plt.close()
    
    # Save histograms for all features
    n_features = trainX.shape[1]  # Total number of features (including Hu Moments)
    fig0, ax0 = plt.subplots(2, (n_features + 1) // 2, figsize=(15, 10))

    # Flatten axis array to simplify plotting in the loop
    ax0 = ax0.flatten()

    for i in range(n_features):
        sns.histplot(trainX[:, i], bins=10, ax=ax0[i], kde=False)
        ax0[i].set_xlabel(gestures.feature_names[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'histogram_all_features.png'))
    plt.close(fig0)  # Close the figure after saving

    # Generate scatter plots of features in the dataset in a for loop
    for i in range(0, trainX.shape[1]):
        for j in range(i + 1, trainX.shape[1]):
            fig1 = plt.figure()
            ax1 = sns.scatterplot(x=trainX[:, i], y=trainX[:, j], hue=le.inverse_transform(trainY))
            ax1.set_title(f"Scatter plot of {gestures.feature_names[i]} and {gestures.feature_names[j]}")
            ax1.set_xlabel(gestures.feature_names[i])
            ax1.set_ylabel(gestures.feature_names[j])
            plt.tight_layout()
            plt.savefig(os.path.join(output_subdir, f'scatter_plot_{gestures.feature_names[i]}_{gestures.feature_names[j]}.png'))
            plt.close(fig1)  # Close the figure after saving
    
    # Save boxplots for all features
    for i in range(0, trainX.shape[1]):
        fig2 = plt.figure()
        ax2 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:,i])
        ax2.set_title(gestures.feature_names[i])
        ax2.set_ylabel(gestures.feature_names[i])
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, 'boxplot_' + gestures.feature_names[i] + '.png'))
        plt.close(fig2)

    # Save feature correlation heatmap
    plt.figure(figsize=(16, 12))
    corr = np.corrcoef(trainX, rowvar=False)
    ax4 = sns.heatmap(corr, annot=True, xticklabels=gestures.feature_names, yticklabels=gestures.feature_names)
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'correlation_heatmap.png'))
    plt.close()
    
    # Close all figures
    plt.close('all')
    
    # Make sure all code is executed before closing
    plt.show(block=True)
    
    print("[INFO] Done exploring features")
    
    # Save end time
    end_time = datetime.datetime.now()
    
    # Calculate and print duration
    duration = end_time - start_time
    print(f"Duration: {duration}")