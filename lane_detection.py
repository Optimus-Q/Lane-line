import cv2
import numpy as np
import argparse


class LaneDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

    def calculate_coordinates(self, img, line_params):
        slope, intercept = line_params
        y1 = img.shape[0]
        y2 = int(y1 * 0.6)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def compute_average_slope_intercept(self, img, lines):
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            params = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = params
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_avg = np.average(left_fit, axis=0)
        right_avg = np.average(right_fit, axis=0)
        left_line = self.calculate_coordinates(img, left_avg)
        right_line = self.calculate_coordinates(img, right_avg)
        return np.array([left_line, right_line])

    def apply_canny(self, image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edge_img = cv2.Canny(blurred_img, 50, 150)
        return edge_img

    def draw_lines(self, img, lines):
        line_img = np.zeros_like(img)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 10)
        return line_img

    def define_roi(self, image):
        img_height = image.shape[0]
        polygon = np.array([
            [(200, img_height), (1100, img_height), (550, 250)]
        ])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygon, 255)
        masked_img = cv2.bitwise_and(image, mask)
        return masked_img

    def process_frame(self, frame):
        canny_img = self.apply_canny(frame)
        roi_img = self.define_roi(canny_img)
        detected_lines = cv2.HoughLinesP(roi_img, 2, np.pi / 180, 100, np.array([]), minLineLength=10, maxLineGap=5)
        avg_lines = self.compute_average_slope_intercept(frame, detected_lines)
        lines_img = self.draw_lines(frame, avg_lines)
        combined_img = cv2.addWeighted(frame, 0.8, lines_img, 1, 1)
        return combined_img

    def detect_lane(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            lane_frame = self.process_frame(frame)
            cv2.imshow('Lane Detection', lane_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Detection on Video")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()

    lane_detector = LaneDetector(args.video_path)
    lane_detector.detect_lane()

