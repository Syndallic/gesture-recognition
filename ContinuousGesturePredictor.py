import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils


def resize_image(image_name):
    basewidth = 100
    img = Image.open(image_name)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(image_name)


def average_background(bg, image, a_weight):
    """
    Runs an accumulating average of the background

    :param bg: The current average background
    :param image: The latest image to update the the background with
    :param a_weight: The accumulation weight - how much the new image affects the running average
    :return: The updated bg
    """
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return bg

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, a_weight)
    return bg


def segment(bg, image, threshold=25):
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(contours) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(contours, key=cv2.contourArea)
        return thresholded, segmented


def get_predicted_class(image):
    # image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([np.expand_dims(gray_image, 2)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))


def show_statistics(predicted_class, confidence):
    text_image = np.zeros((300, 512, 3), np.uint8)
    class_name = str(predicted_class)

    cv2.putText(text_image, "Pedicted Class : " + class_name,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    cv2.putText(text_image, "Confidence : " + str(confidence * 100) + '%',
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)
    cv2.imshow("Statistics", text_image)


def define_model():
    tf.reset_default_graph()
    convnet = input_data(shape=[None, 215, 240, 1], name='input')
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1000, activation='relu')
    convnet = dropout(convnet, 0.75)

    convnet = fully_connected(convnet, 6, activation='softmax')

    convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                         name='regression')

    return tflearn.DNN(convnet, tensorboard_verbose=0)


def load_saved_model():
    model = define_model()
    model.load("TrainedModel/GestureRecogModel.tfl")
    return model


def main():
    # initialize background and weight for running average
    bg = None
    a_weight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0
    start_recording = False

    print("Calibrating...")

    # keep looping, until interrupted
    while True:
        # get the current frame
        grabbed, frame = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 60:
            bg = average_background(bg, gray, a_weight)
            if num_frames == 59:
                print("Calibrated...")
        else:
            # segment the hand region
            hand = segment(bg, gray)

            # check whether hand region is segmented
            if hand is None:
                # if no hand, update running average to accommodate long term lighting changes
                bg = average_background(bg, gray, a_weight / 2)
            else:
                # if yes, unpack the thresholded image and segmented region
                thresholded, segmented = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:
                    # cv2.imwrite('Temp.png', thresholded)
                    # resizeImage('Temp.png')
                    predicted_class, confidence = get_predicted_class(thresholded)
                    show_statistics(predicted_class, confidence)
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
        if keypress == ord("s"):
            start_recording = True


if __name__ == "__main__":
    main()
