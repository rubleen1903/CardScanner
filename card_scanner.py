import cv2
import numpy as np
import sys
import pytesseract

"""============================================================================
PROCEDURE:
    draw_ROI
PARAMETERS:
    img, an image
    corners, an np.array containing a list of four corners/vertices
PURPOSE:
    allows user to drag the circles to redefine the edges
PRODUCES:
    dst, an image with re-drawn circles and lines
============================================================================"""
def draw_ROI(img, corners):

    # Set Colors
    COLOR_1 = (192, 192, 255)
    COLOR_2 = (128, 128, 255)

    # Create a copy of the original image
    cp = img.copy()

    # Draw a circle on each corner
    for pt in corners:
        cv2.circle(cp, tuple(pt), 25, COLOR_1, -1, cv2.LINE_AA)

    # Draw a line that connects the corners
    cv2.line(cp, tuple(corners[0]), tuple(corners[1]), COLOR_2, 2, cv2.LINE_AA)
    cv2.line(cp, tuple(corners[1]), tuple(corners[2]), COLOR_2, 2, cv2.LINE_AA)
    cv2.line(cp, tuple(corners[2]), tuple(corners[3]), COLOR_2, 2, cv2.LINE_AA)
    cv2.line(cp, tuple(corners[3]), tuple(corners[0]), COLOR_2, 2, cv2.LINE_AA)

    # This is for UX - makes the circles and lines a little transparent
    dst = cv2.addWeighted(img, 0.3, cp, 0.7, 0)

    return dst

"""============================================================================
PROCEDURE:
    onMouse
PARAMETERS:
    event, any cv2.MouseEventTypes events
    x, x-coordinate of the event
    y, y-coordinate of the event
    flags, any cv2.MouseEventFlags flags
    param, any additional parameters 
PURPOSE:
    a callback function for mouse events
PRODUCES:
    None - a void function
============================================================================"""
def onMouse(event, x, y, flags, param):
    
    # This is used for identifying change in movement of circles
    global pt_old

    # Unpack the parameters from the redefine() function 
    src_quad2 = param[0]
    drag_src = param[1]
    src = param[2]

    if event == cv2.EVENT_LBUTTONDOWN:  # Left button pressed
        for i in range(4):
            if cv2.norm(src_quad2[i] - (x, y)) < 25:
                drag_src[i] = True
                pt_old = (x, y)
                break

    if event == cv2.EVENT_LBUTTONUP:    # Left button released
        for i in range(4):
            drag_src[i] = False

    if event == cv2.EVENT_MOUSEMOVE:    # Moving / Dragging
        for i in range(4):
            if drag_src[i]:
                dx = x - pt_old[0]
                dy = y - pt_old[1]

                src_quad2[i] += (dx, dy)

                cpy = draw_ROI(src, src_quad2)
                cv2.imshow('Redefine Card', cpy)
                pt_old = (x, y)
                break

"""============================================================================
PROCEDURE:
    capture_screen
PARAMETERS:
    cam, a VideoCapture object
PURPOSE:
    Captures an image from a device camera and saves it as a jpg file
PRODUCES:
    None - a void function
============================================================================"""
def capture_screen(cam):

    # Main loop
    while True:

        key = cv2.waitKey(1)          # Keyboard input 
        ret, frame = cam.read()       # Reading frames from camera

        if not ret:                   # Checking camera input 
            break

        cv2.imshow("Card Scanner", frame)  # Open window

        # Quit window
        if key == ord("q") or key == ord("Q"):
            break

        # Save image
        if key == ord(" "): 
            cv2.imwrite(filename="saved_img.jpg", img=frame)
            cv2.waitKey(500)
            break

    cam.release()
    cv2.destroyAllWindows()

"""============================================================================
PROCEDURE:
    reorderPts
PARAMETERS:
    pts, a numpy.ndarray
PURPOSE:
    Ensures that the four corner-points of a card is always arranged in
    a counter-clockwise fashion 
PRODUCES:
    pts, a reordered numpy.ndarray
============================================================================"""
def reorderPts(pts):

    # Lexsort by x coordinates first and then y coordinates
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[idx]

    # Sawp if y of the first point is larger than that of the second point
    if pts[0, 1] > pts[1, 1]:
        pts[[0, 1]] = pts[[1, 0]]

    # Sawp if y of the third point is larger than that of the fourth point
    if pts[2, 1] < pts[3, 1]:
        pts[[2, 3]] = pts[[3, 2]]

    return pts

"""============================================================================
PROCEDURE:
    display_text
PARAMETERS:
    img, an image
    text, a string value
PURPOSE:
    displays given text at the top-center of an image 
PRODUCES:
    None - a void function
============================================================================"""
def display_text(img, text):

    # Set up text elements
    text_font      = cv2.FONT_HERSHEY_SIMPLEX
    text_color     = (0, 255, 0)
    text_scale     = 0.8
    text_thickness = 2
    text_size = cv2.getTextSize(text, text_font, text_scale, text_thickness)

    # Calculate the text display position
    text_x = int( (img.shape[1] - text_size[0][0]) / 2 )
    text_y = 50   # Giving some top-margin

    # Display text
    cv2.putText(img, text, (text_x, text_y), text_font, text_scale, 
                text_color, text_thickness, cv2.LINE_AA)
    
"""============================================================================
PROCEDURE:
    redefine
PARAMETERS:
    src, a source image
    card_width, standard width of a business card 
    card_height, standard height of a business card 
PURPOSE:
    when program fails to detect card, user redefines the edges
PRODUCES:
    dst, 
============================================================================"""
def redefine(src, card_width, card_height):
    
    # Attain height and width information of source image
    src_h, src_w = src.shape[:2]

    # Set source and destination quadrants
    src_quad2 = np.array([[30, 30], 
                         [30, src_h-30], 
                         [src_w-30, src_h-30],
                         [src_w-30, 30]], np.float32)

    dst_quad2 = np.array([[0, 0], 
                         [0, card_height-1], 
                         [card_width-1, card_height-1], 
                         [card_width-1, 0]], np.float32)

    # Initialize corners drag status
    drag_src = [False, False, False, False]

    # Group elements together to pass in
    param = [src_quad2, drag_src, src]

    # Displaying region of interest
    disp = draw_ROI(src, src_quad2)
    cv2.imshow("Redefine Card", disp)

    # Handle mouse events
    cv2.setMouseCallback("Redefine Card", onMouse, param)

    while True:
        key = cv2.waitKey()

        if key == ord(" "):
            # Attain transformation information and warp perspective
            pers = cv2.getPerspectiveTransform(src_quad2, dst_quad2)
            dst = cv2.warpPerspective(src, pers, (card_width, card_height))

            cv2.destroyWindow("Redefine Card")
            return dst 

        elif key == 27: # ESC Key
            cv2.destroyWindow("Redefine Card")
            sys.exit()


"""============================================================================
PROCEDURE:
    detect_card
PARAMETERS:
    src, a source image
    card_width, standard width of a business card 
    card_height, standard height of a business card 
PURPOSE:
    Takes an input image, detects business card, and cuts out only the card
PRODUCES:
    dst, a new image of only the card in correct persective/dimensions
============================================================================"""
def detect_card(src, card_width, card_height):

    # Set up quadrants for warping perspective
    src_quad = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32)
    dst_quad = np.array([[0, 0], 
                        [0, card_height-1], 
                        [card_width-1, card_height-1], 
                        [card_width-1, 0]], np.float32)

    # Destination to receive final card image
    dst = np.zeros((card_height, card_width), np.uint8)

    # Preprocess - binarize grayscaled image with threshold (Otsu's method)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    _, src_bin = cv2.threshold(src_gray, 0, 255, 
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # _, src_bin = cv2.threshold(src_gray, 150, 255, cv2.THRESH_BINARY)

    # cv2.imshow("src_bin", src_bin)

    # Find contours and detect card in the image
    contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_NONE)

    # Create a copy of source image
    cpy = src.copy()

    for pts in contours:
        # Ignore if contour area seems insignificant
        if cv2.contourArea(pts) < 1000:
            continue

        # Simplify contour lines with Douglas–Peucker algorithm
        approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.02, True)

        # Ignore if non-convex or non-rectangle
        if not cv2.isContourConvex(approx) or len(approx) != 4:
            continue
        
        # Draw lines around the detected card
        cv2.polylines(cpy, [approx], True, (0, 255, 0), 2, cv2.LINE_AA)
        src_quad = reorderPts(approx.reshape(4, 2).astype(np.float32))

    # Attain transformation information and warp perspective
    pers = cv2.getPerspectiveTransform(src_quad, dst_quad)
    dst = cv2.warpPerspective(src, pers, (card_width, card_height))

    display_text(cpy, "Type 'y': continue or 'n': redefine")

    cv2.imshow('Detected Edges', cpy)
    
    cv2.namedWindow("Card", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Card", card_width//2, card_height//2)
    cv2.imshow('Card', dst)

    # Whether or not user wants to redefine.
    redef_status = False

    while True:
        key = cv2.waitKey()

        # If y/Y, then continue on to extract text information
        if key == ord("y") or key == ord("Y"):
            cv2.destroyAllWindows()
            return dst, redef_status

        # If n/N, then redefine card edges
        elif key == ord("n") or key == ord("N"):
            redef_status = True
            cv2.destroyAllWindows()
            return dst, redef_status

"""============================================================================
                                     MAIN
============================================================================"""
def main():

    # Constants
    CARD_WIDTH, CARD_HEIGHT = 720, 400

    # Initialize and pen default camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Check for any errors opening the camera
    if not cap.isOpened():
        print("Error: Camera not opening.")
        sys.exit()

    # Take a photo of a business card
    capture_screen(cap)

    # Attain the saved image
    src = cv2.imread('saved_img.jpg')

    # Check for opening the saved image
    if src is None:
        print("Error: Image not found.")
        sys.exit()
    
    # Detect only the card in the image
    card = detect_card(src, CARD_WIDTH, CARD_HEIGHT)

    if card[1] == True:
        card = redefine(src, CARD_WIDTH, CARD_HEIGHT)
    else:
        card = card[0]
    
    # Attain text information using Tesseract-OCR
    card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    print("="*80, "\n")
    print(pytesseract.image_to_string(card_gray, lang='eng'))
    print("="*80, "\n")

    # Show final card
    cv2.namedWindow("Card", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Card", CARD_WIDTH//2, CARD_HEIGHT//2)
    cv2.imshow('Card', card)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()