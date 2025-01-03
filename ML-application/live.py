import cv2 as cv

from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from classifiers import ML_DecisionTree
from fileHandler import createOutputDir, createSubDir, createTimestampDir, loadFiles
from extract import getFeatures, getLargestContour
from segment import prepareImage
import fetch_data
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def DecisionTree(gestures, coded_labels, label_names, outputDirectory):    
    
        # Best hyperparameters found
    best_params = {
        'criterion': 'entropy',
        'max_depth': None,
        'max_features': 'sqrt',
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'splitter': 'best'
    }

    # Create and train the Decision Tree classifier
    clf = DecisionTreeClassifier(**best_params)
    clf.fit(gestures.data, coded_labels)
    return clf

        
      


if __name__ == "__main__":
    """live """
    cap = cv.VideoCapture(0)
    outputDir = createOutputDir()
    timestampDir = createTimestampDir(outputDir)

    # Retrieve the file list
    fileList = loadFiles()
    gestures = fetch_data.datasetBuilder(fileList, timestampDir)

    # Encode the categorical labels
    le = LabelEncoder()
    coded_labels = le.fit_transform(gestures.target)
    

    # Filter the dataset to include only the top 5 unique features

    modelDir = createSubDir(timestampDir, 'Model performance')
    model = DecisionTree(gestures, coded_labels, le.classes_, modelDir)
    
    
    
    print("[INFO] Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        imageMask = prepareImage(frame)
        contour = getLargestContour(imageMask)

        # Initialize label_with_likelihood to avoid issues if features is None
        
        
        # Extract features from the contour
        features = getFeatures(contour)
     
        if features is not None:
            features = np.array(features).reshape(1, -1)  # Ensure shape is correct
            predicted_label_code = model.predict(features)[0]
            predicted_label = le.inverse_transform([predicted_label_code])[0]
        
        # Get the likelihood if available from your model
        probabilities = model.predict_proba(features)[0]
        predicted_likelihood = probabilities[predicted_label_code]
        label_with_likelihood = f"{predicted_label} ({predicted_likelihood * 100:.2f}%)"

        
        cv.putText(frame, label_with_likelihood, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        # Draw the contour and display prediction
        if contour is not None:
            cv.drawContours(frame, [contour], -1, (0, 0, 0), 2)
        
        cv.imshow("Live Gesture Detection", frame)
        
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv.destroyAllWindows()
    
    
    
    