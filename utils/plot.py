import json
import math
import matplotlib.pyplot as plt
from PIL import Image

def plot_curves(log_path, save_path):
    logs = json.load(open(log_path))

    epochs = [x["epoch"] for x in logs]
    train_loss = [x["train_loss"] for x in logs]
    em = [x["dev_EM"] for x in logs]
    f1 = [x["dev_F1"] for x in logs]

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(epochs, em, label="EM")
    plt.plot(epochs, f1, label="F1")
    plt.legend()
    plt.title("Dev Metrics")

    plt.savefig(save_path)
    plt.show()

def plot_image_predictions(data, save_path, max_cols=3):
    num_images = len(data)
    if num_images == 0:
        return

    cols = min(max_cols, num_images)
    rows = math.ceil(num_images / cols)

    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, item in enumerate(data):
        if isinstance(item, dict):
            image = item.get("image")
            question = item.get("question", "")
            ground_truth = item.get("ground_truth", item.get("answer", ""))
            prediction = item.get("prediction", item.get("predicted_answer", ""))
        else:
            try:
                image, question, ground_truth, prediction = item
            except ValueError:
                continue

        if image is None:
            continue

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.axis("off")

        title = f"Q: {question}\nGT: {ground_truth}\nPred: {prediction}"
        plt.title(title, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

if __name__ == "__main__":
    plot_curves("/home/fit02/dien_workspace/vqa/outputs_ocr/logs.json", "/home/fit02/dien_workspace/vqa/outputs_ocr/logs.png")

    # predictions_path = "/home/fit02/dien_workspace/vqa/outputs/predictions.json"

    # try:
    #     with open(predictions_path, "r") as f:
    #         content = f.read().strip()

    #     if not content:
    #         print(f"predictions.json is empty at: {predictions_path}")
    #         data_new = []
    #     else:
    #         data_new = json.loads(content)
    # except FileNotFoundError:
    #     print(f"predictions.json not found at: {predictions_path}")
    #     data_new = []
    # except json.JSONDecodeError as e:
    #     print(f"Could not decode JSON from {predictions_path}: {e}")
    #     data_new = []

    # if not data_new:
    #     print("No prediction data to plot. Make sure predictions.json is generated and non-empty.")
    # else:
    #     data0 = [data for data in data_new if data.get("EM") == 0]
    #     data1 = [data for data in data_new if data.get("EM") == 1]

    #     for item in data0:
    #         item["image"] = Image.open(item["image_path"]).convert("RGB")

    #     for item in data1:
    #         item["image"] = Image.open(item["image_path"]).convert("RGB")

    #     try:
    #         plot_image_predictions(data0[:6], "/home/fit02/dien_workspace/vqa/outputs/incorrect_predictions.png")
    #         plot_image_predictions(data1[:6], "/home/fit02/dien_workspace/vqa/outputs/correct_predictions.png")
    #     except Exception as e:
    #         print(f"Error in plotting image predictions: {e}")