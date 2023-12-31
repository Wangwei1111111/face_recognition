import argparse
import sys
import time

import cv2

import person

# 视频上方覆盖层：人脸框、名字、帧率
def add_overlays(frame, persons, frame_rate):

    # 若查询到人员信息，在方框下方显示信息
    if persons is not None:
        for person in persons:
            face_bb = person.bounding_box.astype(int)
            cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), (0, 255, 0), 2)
            if person.pid is not None:
                cv2.putText(frame, person.pid, (face_bb[0], face_bb[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    # 显示帧率
    cv2.putText(frame, str(frame_rate) + " fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2,
                lineType=2)


def main(args):
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5    # seconds
    frame_rate = 0
    frame_count = 0

    video_capture = cv2.VideoCapture(0)
    face_recognition = person.Recognition()
    start_time = time.time()


    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            persons = face_recognition.identify(frame)

            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, persons, frame_rate)

        frame_count += 1
        cv2.imshow('Video', frame)

        # Q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
