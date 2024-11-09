import cv2 as cv
from extract import getFeatures, getLargestContour
from segment import prepareImage
import fetch_data
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def live(gestures,le):
    cap = cv.VideoCapture(0)
    encoded_labels = le.fit_transform(gestures.target)
    best_params = {
        'criterion': 'entropy',
        'max_depth': None,
        'max_features': 'sqrt',
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'splitter': 'best'
    }
    model = DecisionTreeClassifier(**best_params)
    model.fit(gestures.data,encoded_labels)
    print("[INFO] Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        imageMask = prepareImage(frame)
        contour = getLargestContour(imageMask)
        # cv.drawContours(frame, contour, -1, (0, 255, 0), 2)
    
        # Get features from contour 
        features = getFeatures(contour)
        
          
        if features is not None:
            # Ensure the features are reshaped into a 2D array as required by the model
            features = np.array(features).reshape(1, -1)  # Reshaping to (1, n_features)
            
            # Predict the label based on the features
            predicted_label_code = model.predict(features)[0]
            predicted_label = le.inverse_transform([predicted_label_code])[0]
            probabilities = model.predict_proba(features)[0]
            predicted_likelihood = probabilities[predicted_label_code]
            label_with_likelihood = f"{predicted_label} ({predicted_likelihood*100:.2f}%)"
        if contour is not None:
            cv.drawContours(frame, [contour], -1, (0, 0, 0), 2) 
        cv.putText(frame, label_with_likelihood, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow("Live Gesture Detection",frame)

        
        if cv.waitKey(1) & 0xFF == ord("q"):
            break