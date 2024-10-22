import imutils
import cv2 

image = cv2.imread("c:\\Users\\koend\\Documents\\steenpapierschaar-tech\\ImagerTool\\data\\paper\\1727519866943305906.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Pas een drempelwaarde toe
rect, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Zoek de contouren met een hiërarchie, zodat we zowel de buitenste als de binnenste contouren hebben
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Teken alle contouren (zowel buiten als binnen)
for i, contour in enumerate(contours):
    # Bepaal de omtrek van de contour
    perimeter = cv2.arcLength(contour, True)
    
    # Print of het een buitenste of binnenste contour is
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  

# Toon het resultaat
cv2.imshow('Contour Image', image)
cv2.imshow('Threshold', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()