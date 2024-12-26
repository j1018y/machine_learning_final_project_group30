import os
import re
import pandas as pd

def extract_metrics(file_path):
    """
    从单个 txt 文件中提取 Error Rate、Recall、Precision 和 F1-score。
    """
    metrics = {"error_rate": None, "recall": None, "precision": None, "f1_score": None}
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if "Error Rate:" in line:
                metrics["error_rate"] = float(re.search(r"Error Rate: (\d+\.\d+)%", line).group(1))
            elif "Recall:" in line:
                metrics["recall"] = float(re.search(r"Recall: (\d+\.\d+)", line).group(1))
            elif "Precision:" in line:
                metrics["precision"] = float(re.search(r"Precision: (\d+\.\d+)", line).group(1))
            elif "F1-score:" in line:
                metrics["f1_score"] = float(re.search(r"F1-score: (\d+\.\d+)", line).group(1))
    return metrics

def analyze_confidence_by_version(results_folder):
    """
    分析每个版本下，每个 confidence 对应的所有人 F1-score。
    """
    all_versions_data = []  # 用于保存所有版本数据
    summary_data = []  # 用于保存汇总数据

    # 遍历固定范围的版本号，从 ver0 到 ver10
    for version_num in range(11):  # 0 到 10
        version_folder = f"ver{version_num}"
        version_path = os.path.join(results_folder, version_folder)

        if not os.path.exists(version_path) or not os.path.isdir(version_path):
            print(f"Skipping missing or invalid directory: {version_folder}")
            continue

        print(f"Processing {version_folder}...")

        version_data = {"version": version_folder, "confidence_metrics": {}}  # 记录版本数据

        # 初始化 confidence_metrics 的结构
        for confidence in range(10, 151, 10):
            version_data["confidence_metrics"][confidence] = {}

        # 遍历每个人的结果文件夹
        for person_folder in os.listdir(version_path):
            person_path = os.path.join(version_path, person_folder)
            if not os.path.isdir(person_path):
                print(version_folder)
                continue

            # 遍历每个 confidence 的结果文件
            for result_file in os.listdir(person_path):
                result_path = os.path.join(person_path, result_file)
                if result_file.endswith(".txt"):
                    confidence_match = re.search(r"conf_(\d+)", result_file)
                    if confidence_match:
                        confidence = int(confidence_match.group(1))
                        metrics = extract_metrics(result_path)

                        # 保存当前人的 F1-score
                        version_data["confidence_metrics"][confidence][person_folder] = metrics["f1_score"]

        # 计算每个 confidence 的总平均 F1-score
        confidence_summary = {}
        for confidence, person_scores in version_data["confidence_metrics"].items():
            if person_scores:
                total_f1_score = sum(score for score in person_scores.values() if score is not None)
                confidence_summary[confidence] = total_f1_score / len(person_scores)
            else:
                confidence_summary[confidence] = 0.0  # 如果没有数据，设置为 0

        # 找出总平均 F1-score 最高的 confidence
        best_confidence = max(confidence_summary, key=confidence_summary.get)
        summary_data.append({
            "version": version_folder,
            "best_confidence": best_confidence,
            "average_f1_score": confidence_summary[best_confidence]
        })

        all_versions_data.append(version_data)

    return all_versions_data, summary_data

def save_version_analysis(all_versions_data, summary_data, output_folder):
    """
    保存每个版本的分析结果以及汇总数据到文件。
    """
    # 保存每个版本的详细数据
    for version_data in all_versions_data:
        version_file = os.path.join(output_folder, f"{version_data['version']}_analysis.txt")
        with open(version_file, "w", encoding="utf-8") as f:
            f.write(f"Version: {version_data['version']}\n")
            f.write("=" * 80 + "\n")

            for confidence, person_scores in version_data["confidence_metrics"].items():
                f.write(f"Confidence: {confidence}\n")
                f.write("-" * 80 + "\n")
                for person, f1_score in person_scores.items():
                    f.write(f"{person:<20}: {f1_score if f1_score is not None else 'N/A'}\n")
                f.write("\n")

    # 保存汇总数据
    summary_file = os.path.join(output_folder, "version_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Version Summary\n")
        f.write("=" * 80 + "\n")
        for summary in summary_data:
            f.write(f"Version: {summary['version']}\n")
            f.write(f"Best Confidence: {summary['best_confidence']}\n")
            f.write(f"Average F1-score: {summary['average_f1_score']:.2f}\n")
            f.write("-" * 80 + "\n")

if __name__ == "__main__":
    # 指定结果文件夹路径
    results_folder = "version_results_for_old_data"
    output_folder = "version_analysis_f1_for_old_data"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 分析所有版本
    all_versions_data, summary_data = analyze_confidence_by_version(results_folder)

    # 保存分析结果
    save_version_analysis(all_versions_data, summary_data, output_folder)

    print("\nAnalysis completed. Results saved in:", output_folder)
