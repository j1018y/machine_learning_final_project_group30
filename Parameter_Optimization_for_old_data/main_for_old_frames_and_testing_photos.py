import cv2
import numpy as np
import pandas as pd
from itertools import product
import os
from pathlib import Path


def save_aligned_results(df, file_path, columns, version_info, error_rate, recall, precision, f1_score, tp, fp, tn, fn):
    """
    保存对齐结果到文件
    """
    with open(file_path, "w", encoding="utf-8") as f:
        # 空行
        f.write("\n")

        # 写入版本信息
        f.write(f"Frame Count: {version_info['frames']}\n")
        f.write(f"LBPH Parameters: radius={version_info['lbph_params']['radius']}, "
                f"neighbors={version_info['lbph_params']['neighbors']}, "
                f"grid_x={version_info['lbph_params']['grid_x']}, "
                f"grid_y={version_info['lbph_params']['grid_y']}\n")

        # 空行
        f.write("\n")

        # 写入标题行
        header = "".join(f"{col:<15}" for col in columns)
        f.write(header + "\n")
        f.write("=" * len(header) + "\n")

        # 写入每一行数据
        for _, row in df.iterrows():
            line = "".join(f"{str(row[col]):<15}" for col in columns)
            f.write(line + "\n")

        # 空行
        f.write("\n")

        # 写入错误率及其他指标
        f.write(f"Error Rate: {error_rate:.2f}%\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"F1-score: {f1_score:.2f}\n")
        f.write(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")


# 初始化配置
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

photo_number = 38
photo_to_person = [
    "LHW", "LHW", "LHW", "LHW", "LYD", "Face", "Face", "GGL", "GGL", "GGL",
    "GGL", "GGL", "Leo", "LJF", "Boson", "Boson", "Boson", "Boson", "Kuan", "Kuan",
    "Kuan", "Lu", "A3o", "A3o", "A3o", "A3o", "A3o", "LZN", "LZN", "Yu",
    "LZNB", "YCL", "YHS", "LZNM", "Yu", "Yu", "Deng", "Girl1"
]

frame_counts = [100, 200, 300, 400, 500]
lbph_params = [
    {"radius": 1, "neighbors": 8, "grid_x": 8, "grid_y": 8},
    {"radius": 2, "neighbors": 10, "grid_x": 10, "grid_y": 10}
]
confidence_thresholds = list(range(10, 151, 10))  # 10 到 150，每次增加 10

training_names = [
    "Kuan", "Boson", "LZNB", "YCL", "A3o",
    "GGL", "LJF", "LYD", "Leo", "Face", 
    "LHW", "YHS", "Girl1", "Lu", "Deng", 
    "DN", "Yu", "ZZJ", "LZN", "LZNM"
]

output_folder = "version_results_for_old_data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

version = 1
start_version = 6  # 从第5个版本开始
for frame_count, lbph_param in product(frame_counts, lbph_params):
    if version < start_version:
        version += 1
        continue  # 跳过前四个版本
    print(f"\nTraining with frames={frame_count}, "
          f"LBPH(radius={lbph_param['radius']}, neighbors={lbph_param['neighbors']}, "
          f"grid_x={lbph_param['grid_x']}, grid_y={lbph_param['grid_y']})...")

    version_folder = os.path.join(output_folder, f"ver{version}")
    if not os.path.exists(version_folder):
        os.makedirs(version_folder)

    for vid_id, name in enumerate(training_names):
        print(f"\nProcessing {name} (vid_{vid_id})...")

        faces = []
        ids = []
        frame_paths = list(Path(f'old_frames/vid_{vid_id}').glob('*.jpg'))
        if len(frame_paths) == 0:
            print(f"No frames found for vid_{vid_id}. Skipping...")
            continue

        while len(frame_paths) < frame_count:
            frame_paths += frame_paths[:frame_count - len(frame_paths)]

        for frame_path in frame_paths[:frame_count]:
            img = cv2.imread(str(frame_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_detected = detector.detectMultiScale(gray)

            for (x, y, w, h) in faces_detected:
                face_region = gray[y:y + h, x:x + w]
                faces.append(face_region)
                ids.append(vid_id + 1)
                

        if len(faces) == 0:
            print(f"No faces detected for {name}. Skipping...")
            continue

        print(f"\nTraining model for {name}...")
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=lbph_param['radius'],
            neighbors=lbph_param['neighbors'],
            grid_x=lbph_param['grid_x'],
            grid_y=lbph_param['grid_y']
        )
        recognizer.train(faces, np.array(ids))

        model_path = os.path.join(version_folder, f"{name}.yml")
        recognizer.save(model_path)
        print(f"Model saved as {model_path}.")

    for name in training_names:
        person_result_folder = os.path.join(version_folder, f"results_for_{name}")
        if not os.path.exists(person_result_folder):
            os.makedirs(person_result_folder)

        model_path = os.path.join(version_folder, f"{name}.yml")
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}, skipping {name}.")
            continue

        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=lbph_param['radius'],
            neighbors=lbph_param['neighbors'],
            grid_x=lbph_param['grid_x'],
            grid_y=lbph_param['grid_y']
        )
        recognizer.read(model_path)

        for confidence_threshold in confidence_thresholds:
            records = []
            tp, fp, tn, fn = 0, 0, 0, 0

            for ID in range(1, photo_number + 1):
                img = cv2.imread(f'old_testing_photos/{ID}.jpg')
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(40, 40))
                predicted_answer = "Unknown"
                confidence = None

                for (x, y, w, h) in faces:
                    face_region = gray[y:y + h, x:x + w]
                    idnum, confidence = recognizer.predict(face_region)
                    if confidence < confidence_threshold:
                        predicted_answer = training_names[idnum - 1]

                actual_person = photo_to_person[ID - 1]

                if predicted_answer == actual_person:
                    result = "Right"
                    tp += 1
                elif predicted_answer == "Unknown" and actual_person != name:
                    result = "Right"
                    tn += 1
                else:
                    result = "Wrong"
                    if predicted_answer == "Unknown":
                        fn += 1
                    else:
                        fp += 1

                records.append({
                    "照片编号": ID,
                    "实际人名": actual_person,
                    "预测人名": predicted_answer,
                    "Confidence": f"{confidence:.2f}" if confidence else "N/A",
                    "Result": result
                })

            result_df = pd.DataFrame(records)
            wrong_count = sum(result_df['Result'] == "Wrong")
            total_count = len(result_df)
            error_rate = (wrong_count / total_count) * 100 if total_count > 0 else 0

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            result_file = os.path.join(person_result_folder, f"results_conf_{confidence_threshold}.txt")
            save_aligned_results(
                result_df,
                result_file,
                ["照片编号", "实际人名", "预测人名", "Confidence", "Result"],
                {"frames": frame_count, "lbph_params": lbph_param},
                error_rate,
                recall,
                precision,
                f1_score,
                tp,
                fp,
                tn,
                fn
            )

            print(f"Results for {name} with confidence={confidence_threshold} saved to {result_file}.")
            print(f"Error Rate: {error_rate:.2f}%, Recall: {recall:.2f}, Precision: {precision:.2f}, F1-score: {f1_score:.2f}")

    version += 1

print("\nAll combinations processed!")
