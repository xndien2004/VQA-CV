import json
import matplotlib.pyplot as plt

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

def plot_image_predictions(data, save_path):
    num_images = len(data)
    plt.figure(figsize=(15, 5 * num_images))

    for i, (image, question, ground_truth, prediction) in enumerate(data):
        plt.subplot(num_images, 1, i + 1)
        plt.imshow(image)
        plt.axis('off')
        title = f"Q: {question}\nGT: {ground_truth} | Pred: {prediction}"
        plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()