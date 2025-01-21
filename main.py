import os
import cv2
import numpy as np
import pickle

class Interpreter:
    subject_id_map = {
        "21": "math",
        "22": "physics",
        "23": "tpat3",
        "11": "pretest",
        "12": "posttest"
    }
    answer_key = {
        "tpat3": [
            "3","1","2","4","1","5","2","3","2","4","3","4","1","5","3",
            "5","4","2","1","2","5","1","3","1","5","3","4","4","3","1",
            "4","2","4","3","5","1","4","3","5","2","1","1","4","5","3",
            "3","5","1","2","1","1","5","2","4","2","1","3","4","3","2",
            "1","3","1","5","5","2","4","4","3","1"
        ],
        "math": [
            "3","4","1","1","1","3","5","5","4","5","5","5","2","2","4","4","1","3","2","4","3","3","1","2","2",
            "000200", "000600", "000050", "006250", "001500"
        ],
        "physics": [
            "5","1","2","1","4","3","1","2","2","4","1","4","3","1","2","2","5","4","1","1","1","3","5","5","3",
            "003001", "013720", "010000", "025000", "005760"
        ],
    }

    @staticmethod
    def top(blacken: dict[str, list[int]]) -> tuple[str, str, str, bool]:
        subject = ["X"] * 2
        for idx, black in enumerate(blacken["top"][:20]):
            if not black:
                continue
            if subject[idx // 10] == "X":
                subject[idx // 10] = str(idx % 10)
            else:
                subject_id = ["ERR"]
                break
        subject = "".join(subject)
        venue = "ERR"
        if subject in Interpreter.subject_id_map:
            subject = Interpreter.subject_id_map[subject]
        if subject in ["math", "physics", "tpat3"]:
            venue = "FEST"
        elif subject in ["pretest", "posttest"]:
            venue = "FE"

        # Extract student ID
        student_id = ["X"] * 8
        for idx, black in enumerate(blacken["top"][20:100]):
            if not black:
                continue
            if student_id[idx // 10] == "X":
                student_id[idx // 10] = str(idx % 10)
            else:
                student_id = ["ERR"]
                break
        if not AnswerSheet.valid_id(list(map(int, student_id))):
            student_id = ["IVD"]
        student_id = "".join(student_id)
        
        # Extract cancel
        cancel = bool(blacken["top"][100])

        return (subject, venue, student_id, cancel)
    
    @staticmethod
    def answer(blacken: dict[str, list[int]], subject: str) -> tuple[list[str], list[int], float]:
        answer = None
        if subject in ["math", "physics"]:
            answer = ["X"] * 25 + ["XXXXXX"] * 5
            # Multiple choice section
            for idx, black in enumerate(blacken["answer"][:25*5]):
                if not black:
                    continue
                if answer[idx // 5] == "X":
                    answer[idx // 5] = str(idx % 5 + 1)
                else:
                    answer[idx // 5] = "ERR"
            # Cloze section
            for idx, black in enumerate(blacken["answer"][25*5:]):
                if not black:
                    continue
                question = idx // 60 + 25
                digit = (idx % 60) // 10
                value = idx % 10
                if answer[question][digit] == "X":
                    answer[question] = answer[question][:digit] + str(value) + answer[question][digit + 1:]
                else:
                    self.answer[question] = "ERR"
        elif subject == "tpat3":
            answer = ["X"] * 70
            # Multiple choice section
            for idx, black in enumerate(blacken["answer"]):
                if not black:
                    continue
                if answer[idx // 5] == "X":
                    answer[idx // 5] = str(idx % 5 + 1)
                else:
                    answer[idx // 5] = "ERR"

        # Check answer
        score = []
        for idx, (a, k) in enumerate(zip(answer, Interpreter.answer_key[subject])):
            if a == k:
                score.append(1)
            else:
                score.append(0)

        # Calculate total score
        total_score = 0
        if subject in ["math", "physics"]:
            total_score = sum(score[:25]) * 3 + sum(score[25:]) * 5
        elif subject == "tpat3":
            total_score += sum(score[:15]) * 20 / 15
            total_score += sum(score[15:30]) * 20 / 15
            total_score += sum(score[30:45]) * 20 / 15
            total_score += sum(score[45:60]) * 20 / 15
            total_score += sum(score[60:70]) * 20 / 10
            total_score = round(total_score, 2)

        return (answer, score, total_score)

class AnswerSheet:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.binary_img = None
        self.sections = None
        self.blacken = {
            "top": [],
            "answer": []
        }

        self.subject = None
        self.venue = None
        self.student_id = None
        self.cancel = None
        self.answer = None
        self.score = None
        self.total_score = 0

    def get_file_name(self):
        return os.path.basename(self.img_path)

    def read_dots(self, pickle_name: dict[str, str]) -> None:
        if not self.sections:
            raise Exception("The sections are not cropped")
        for section in pickle_name:
            marks = []
            with open(f'./pickles/{pickle_name[section]}.pkl', "rb") as f:
                marks = pickle.load(f)
            blacken = []
            img = self.sections[section]
            height, width = img.shape
            for mark in marks:
                (x, y), rect_size = mark
                roi = img[
                    int(y * height):int(y * height + rect_size * height),
                    int(x * width):int(x * width + rect_size * height)
                ]
                roi_area = roi.shape[0] * roi.shape[1]
                roi_area = roi.shape[0] * roi.shape[1]
                white = np.sum(roi == 255)
                white_percentage = white / roi_area
                if white_percentage > 0.6:
                    blacken.append(1)
                else:
                    blacken.append(0)
            self.sections[section] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for idx, black in enumerate(blacken):
                (x, y), rect_size = marks[idx]
                if black:
                    cv2.rectangle(self.sections[section], (int(x * width), int(y * height)), (int(x * width + rect_size * height), int(y * height + rect_size * height)), (0, 0, 255), 2)
                else:
                    cv2.rectangle(self.sections[section], (int(x * width), int(y * height)), (int(x * width + rect_size * height), int(y * height + rect_size * height)), (0, 255, 0), 2)
            self.blacken[section] = blacken
    
    @staticmethod
    def valid_id(code: list[int]) -> bool:
        sum = 0 
        payload = code[:-1]
        parity = (len(payload) - 1) % 2 
        for i in range(len(payload)): 
            digit = payload[i] 
            if i % 2 == parity: 
                digit *= 2 
            sum += (digit // 10) + (digit % 10) 
        return (10 - sum % 10) % 10 == code[-1]

    @staticmethod
    def convert_to_binary(img: np.ndarray, thresh: int) -> np.ndarray:
        if len(img.shape) != 3:
            raise Exception("The image is not colored")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def morph_erode(binary_img: np.ndarray, kernel_size: tuple[int, int], iterations: int) -> np.ndarray:
        if len(np.unique(binary_img)) != 2:
            raise Exception("The image is not binary")
        kernel = np.ones(kernel_size, np.uint8)
        return cv2.erode(binary_img, kernel, iterations=iterations)
    
    @staticmethod
    def morph_dilate(binary_img: np.ndarray, kernel_size: tuple[int, int], iterations: int) -> np.ndarray:
        if len(np.unique(binary_img)) != 2:
            raise Exception("The image is not binary")
        kernel = np.ones(kernel_size, np.uint8)
        return cv2.dilate(binary_img, kernel, iterations=iterations)

    @staticmethod
    def get_contours(binary_img: np.ndarray) -> list[np.ndarray]:
        if len(np.unique(binary_img)) != 2:
            raise Exception("The image is not binary")
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    @staticmethod
    def perspective_transform(img: np.ndarray, points: list[list[int]]) -> np.ndarray:
        max_width = max(
            int(np.sqrt(np.power(points[0][0] - points[1][0], 2) + np.power(points[0][1] - points[1][1], 2))),
            int(np.sqrt(np.power(points[2][0] - points[3][0], 2) + np.power(points[2][1] - points[3][1], 2)))
        )
        max_height = max(
            int(np.sqrt(np.power(points[1][0] - points[2][0], 2) + np.power(points[1][1] - points[2][1], 2))),
            int(np.sqrt(np.power(points[3][0] - points[0][0], 2) + np.power(points[3][1] - points[0][1], 2)))
        )
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(np.array(points, dtype="float32"), dst)
        warped = cv2.warpPerspective(img, M, (max_width, max_height))
        return warped

    @staticmethod
    def sort_points(points: list[list[int]]) -> list[list[int]]:
        points = sorted(points, key=lambda x: (x[1], x[0]))
        if points[0][0] > points[1][0]:
            points[0], points[1] = points[1], points[0]
        if points[2][0] < points[3][0]:
            points[2], points[3] = points[3], points[2]
        return points

    @staticmethod
    def crop(binary_img: np.ndarray) -> dict[str, np.ndarray]:
        if len(np.unique(binary_img)) != 2:
            raise Exception("The image is not binary")
        contours = AnswerSheet.get_contours(binary_img)
        height, width = binary_img.shape
        filtered_rect = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) != 4 or cv2.contourArea(contour) < 0.15 * height * width:
                continue
            filtered_rect.append(approx)

        filtered_rect = list(map(lambda x: x.reshape(4, 2), filtered_rect))
        filtered_rect = list(map(lambda x: AnswerSheet.sort_points(x), filtered_rect))
        filtered_rect = list(map(lambda x: AnswerSheet.perspective_transform(binary_img, x), filtered_rect))
        filtered_rect.sort(key=lambda x: x.shape[0] * x.shape[1])
        return {
            "top": filtered_rect[0],
            "answer": filtered_rect[1]
        }

    @staticmethod
    def show_image(img: np.ndarray) -> None:
        cv2.imwrite("_temp.jpg", img)
        os.system("xdg-open _temp.jpg")
        input("Press Enter to continue...")
        os.remove("_temp.jpg")

def evaluate(sheet: AnswerSheet, subject: str) -> None:
    if subject in ["math", "physics"]:
        subject = "alevel"
    sheet.binary_img = AnswerSheet.convert_to_binary(sheet.img, 200)
    sheet.binary_img = AnswerSheet.morph_erode(sheet.binary_img, (2, 2), 1)
    sheet.binary_img = AnswerSheet.morph_dilate(sheet.binary_img, (2, 2), 2)
    sheet.sections = AnswerSheet.crop(sheet.binary_img)
    sheet.read_dots({
        "top": "FEST_top",
        "answer": f"FEST_{subject}_answer"
    })
    sheet.subject, sheet.venue, sheet.student_id, sheet.cancel = Interpreter.top(sheet.blacken)
    print(f"Subject: {sheet.subject} ({sheet.venue}) - Student ID: {sheet.student_id} - Cancel: {sheet.cancel}", end=" - ")
    if sheet.subject == "ERR" or sheet.student_id == "ERR":
        return
    sheet.answer, sheet.score, sheet.total_score = Interpreter.answer(sheet.blacken, sheet.subject)
    print(f"Total score: {sheet.total_score}")

if __name__ == "__main__":
    # Example on how to evaluate an answer sheet
    evaluate(AnswerSheet("./scans/FEST_math/sample_01.jpg"), "math")
    evaluate(AnswerSheet("./scans/FEST_physics/sample_01.jpg"), "physics")
    evaluate(AnswerSheet("./scans/FEST_tpat3/sample_01.jpg"), "tpat3")
