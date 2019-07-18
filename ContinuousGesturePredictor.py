import os

import cv2
import imutils
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


def average_background(bg, image, a_weight):
    """
    Runs an accumulating average of the background

    :param bg: The current average background
    :param image: The latest image to update the the background with
    :param a_weight: The accumulation weight - how much the new image affects the running average
    :return: The updated bg
    """
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


def record_training_image(new_class, num_recorded_frames, image):
    """Saves image to disk and returns whether recording is finished or not"""
    if num_recorded_frames >= 1100:
        print("Recording complete")
        return True

    subfolder = "train/" if num_recorded_frames < 1000 else "test/"
    path = "dataset/{}/{}/{}.png".format(new_class, subfolder, str(num_recorded_frames))

    if cv2.imwrite(path, image):
        print("Saved image #{} out of {}".format(str(num_recorded_frames), 1100))
    else:
        print("Error saving image #{}".format(str(num_recorded_frames)))

    return False


def get_predicted_class(model, image):
    prediction = model.predict([np.expand_dims(image, 2)])
    # return most confident prediction, and confidence level
    return np.argmax(prediction), np.amax(prediction)


def show_statistics(predicted_class, confidence):
    text_image = np.zeros((300, 512, 3), np.uint8)
    class_name = str(predicted_class)

    cv2.putText(text_image, "Predicted Class : " + class_name,
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

    camera = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    num_calibration_frames = 0
    num_recorded_frames = 0
    start_predicting = False
    start_recording = False
    new_class = None
    model = load_saved_model()

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    while True:
        grabbed, frame = camera.read()
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to gray-scale and blur it
        gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

        if num_calibration_frames < 60:
            if num_calibration_frames == 0:
                print("Calibrating...")
            bg = average_background(bg, gray_image, a_weight)
            num_calibration_frames += 1
            if num_calibration_frames == 60:
                print("Calibrated.")
        else:
            hand = segment(bg, gray_image)

            # check whether hand region is found
            if hand is not None:
                # if yes, unpack the thresholded image and segmented region
                thresholded, segmented = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

                if start_predicting:
                    predicted_class, confidence = get_predicted_class(model, thresholded)
                    show_statistics(predicted_class, confidence)

                elif start_recording:
                    # create folder structure for new class
                    if num_recorded_frames == 0:
                        new_class = str(input("Please enter new class name:"))
                        os.makedirs('dataset/{}/train'.format(new_class))
                        os.makedirs('dataset/{}/test'.format(new_class))
                        start_recording = False
                        print("Press 'r' again to start recording")

                    done = record_training_image(new_class, num_recorded_frames, thresholded)
                    num_recorded_frames += 1
                    if done:
                        break

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

        elif keypress == ord("p"):
            start_predicting = True
            show_statistics("No hand", "")

        elif keypress == ord("r"):
            start_recording = True

        elif keypress == ord("c"):
            num_calibration_frames = 0
            start_predicting = False
            start_recording = False
            cv2.destroyWindow("Statistics")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
