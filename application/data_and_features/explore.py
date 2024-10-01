import os
import datetime
import numpy as np
from fetch_data import fetch_data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    """feature exploration"""
    
    # Create output directory with a timestamped subfolder
    output_dir = 'output'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_subdir = os.path.join(output_dir, timestamp)    
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir, exist_ok=True)

    #Features
    area = 0
    contour = 1
    convexHullLength = 2
    convexityDefects = 3

    # get the data path
    data_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'photo_dataset')

    # fetch the data
    gestures = fetch_data(data_path)

    # encode the categorical labels
    le = LabelEncoder()
    coded_labels = le.fit_transform(gestures.target)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(gestures.data, coded_labels,
    test_size=0.25, stratify=gestures.target)#, random_state=42)

    # Data preparation (note that a pipeline  would help here)
    trainX = StandardScaler().fit_transform(trainX)

    # show target distribution
    ax = sns.countplot(x=trainY, color="skyblue")
    ax.set_xticklabels(gestures.unique_targets)
    ax.set_title(data_path + ' count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'target_distribution.png'))

    # Save all features in a single histogram
    fig0, ax0 = plt.subplots(2, 3)
    sns.histplot(trainX[:,0], color="skyblue", bins=10, ax=ax0[0,0])
    sns.histplot(trainX[:,1], color="olive", bins=10, ax=ax0[0,1])#, axlabel=gestures.feature_names[1])
    sns.histplot(trainX[:,2], color="gold", bins=10, ax=ax0[1,0])#, axlabel=gestures.feature_names[2])
    sns.histplot(trainX[:,3], color="teal", bins=10, ax=ax0[1,1])#, axlabel=gestures.feature_names[3])
    ax0[0,0].set_xlabel(gestures.feature_names[0])
    ax0[0,1].set_xlabel(gestures.feature_names[1])
    ax0[1,0].set_xlabel(gestures.feature_names[2])
    ax0[1,1].set_xlabel(gestures.feature_names[3])
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'histogram_all_features.png'))

    #Features
    area = 0
    contour = 1
    convexHullLength = 2
    convexityDefects = 3
    compactness = 4

    # Generate scatter plots of features in the dataset in a for loop
    for i in range(0, trainX.shape[1]):
        for j in range(i+1, trainX.shape[1]):
            fig1 = plt.figure()
            ax1 = sns.scatterplot(x=trainX[:,i], y=trainX[:,j], hue=le.inverse_transform(trainY))
            ax1.set_title("Scatter plot of " + gestures.feature_names[i] + " and " + gestures.feature_names[j])
            ax1.set_xlabel(gestures.feature_names[i])
            ax1.set_ylabel(gestures.feature_names[j])
            plt.tight_layout()
            plt.savefig(os.path.join(output_subdir, 'scatter_plot_' + gestures.feature_names[i] + '_' + gestures.feature_names[j] + '.png'))
    
    # Save boxplots for all features
    for i in range(0, trainX.shape[1]):
        fig2 = plt.figure()
        ax2 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:,i])
        ax2.set_title(gestures.feature_names[i])
        ax2.set_ylabel(gestures.feature_names[i])
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, 'boxplot_' + gestures.feature_names[i] + '.png'))

    # Save feature correlation heatmap
    plt.figure()
    corr = np.corrcoef(trainX, rowvar=False)
    ax4 = sns.heatmap(corr, annot=True, xticklabels=gestures.feature_names, yticklabels=gestures.feature_names)
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'correlation_heatmap.png'))
    
    # Close all figures
    plt.close('all')
    
    # Make sure all code is executed before closing
    plt.show(block=True)
    
    
    print("[INFO] Done exploring features")