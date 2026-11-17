import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
import numpy as np
import brevitas.export as bo
import os
import matplotlib.pyplot as plt


class KWS_Float(Module):
    def __init__(self):
        super(KWS_Float, self).__init__()

        # Model definition: Convolutional architecture for (1, 10, 49) spectrogram features
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )

        # After two (2, 2) max pools on (10, 49): shape becomes (64, 2, 12) = 1536
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2 * 12, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # Reshape to (batch_size, 1, 10, 49)
        x = x.view(-1, 1, 10, 49)
        # Normalize from [0, 1] to [-1, 1]
        x = 2.0 * x - 1.0
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantized model for keyword spotting")
    parser.add_argument("filename", type=str, nargs="?", default="kws_float32", help="Output file name")
    parser.add_argument(
        "--epochs", type=int, nargs="?", default=15, help="Number of epochs"
    )
    parser.add_argument(
        "--gpu", type=str, nargs="?", default=0, help="Index of GPU if available"
    )
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--seed", type=int, nargs="?", default=5, help="Random seed")
    parser.add_argument(
        "--lr", type=float, nargs="?", default=0.001, help="Learning rate"
    )
    parser.add_argument("--bsize", type=int, nargs="?", default=256, help="Batchsize")

    args = parser.parse_args()
    bsize = args.bsize
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs("models", exist_ok=True)

    # Dataset parameter
    tf_dataset_labels = [
        "unknown",
        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes",
    ]

    # class for preprocessed KWS data
    from torchvision import transforms
    from speechcommands_dataset import SpeechCommandsDataset
    from sklearn.model_selection import train_test_split
    import numpy as np

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)
    model = KWS_Float().to(device)

    # Load and split training data
    data = np.load("data/kws_train_X.npy")
    labels = np.load("data/kws_train_Y.npy")
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.2, random_state=random_seed
    )
    train_set = SpeechCommandsDataset(
        X_train, y_train, transform=transforms.Compose([transforms.ToTensor()])
    )
    val_set = SpeechCommandsDataset(
        X_val, y_val, transform=transforms.Compose([transforms.ToTensor()])
    )

    # Load test data
    data_test = np.load("data/kws_test_X.npy")
    labels_test = np.load("data/kws_test_Y.npy")
    test_set = SpeechCommandsDataset(
        data_test, labels_test, transform=transforms.Compose([transforms.ToTensor()])
    )

    # Encapsulate data into dataloader form
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=bsize, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=bsize, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=bsize, shuffle=False
    )

    # Standard CrossEntropyLoss for speech commands classification
    error = torch.nn.CrossEntropyLoss().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.train:
        highestAcc = 0.0
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        print(f"Starting training for {args.epochs} epochs with batch size {bsize} and lr {args.lr}...")

        for epoch in range(args.epochs):
            model.train()
            error.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                prediction = model.forward(images)
                loss = error(prediction, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                pred = prediction.data.argmax(1, keepdim=True)
                train_correct += pred.eq(labels.data.view_as(pred)).sum().item()
                train_total += images.size(0)

            scheduler.step()

            train_loss /= train_total
            train_acc = train_correct * 100.0 / train_total
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)

            # Validation
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            model.eval()
            error.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    prediction = model.forward(images)
                    loss = error(prediction, labels)

                    val_loss += loss.item() * images.size(0)
                    pred = prediction.data.argmax(1, keepdim=True)
                    val_correct += pred.eq(labels.data.view_as(pred)).sum().item()
                    val_total += images.size(0)

            val_loss /= val_total
            val_acc = val_correct * 100.0 / val_total
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            # Save the model with highest accuracy
            if val_acc > highestAcc:
                highestAcc = val_acc
                torch.save(obj=model.state_dict(), f="models/" + args.filename + ".pt")

            print(
                f"Epoch: {epoch:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% (Best: {highestAcc:.2f}%)"
            )

        # Save training and validation history
        history = {
            "train_loss": train_loss_list,
            "train_acc": train_acc_list,
            "val_loss": val_loss_list,
            "val_acc": val_acc_list,
        }
        np.save("models/" + args.filename + "_history.npy", history)

        # Visualize accuracy and loss curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(range(1, args.epochs + 1), train_acc_list, label="Train Accuracy", marker="o")
        ax1.plot(range(1, args.epochs + 1), val_acc_list, label="Val Accuracy", marker="s")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Training & Validation Accuracy")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(range(1, args.epochs + 1), train_loss_list, label="Train Loss", marker="o")
        ax2.plot(range(1, args.epochs + 1), val_loss_list, label="Val Loss", marker="s")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Training & Validation Loss")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plot_path = "models/" + args.filename + "_accuracy.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Accuracy plot saved to {plot_path}")

        # Brevitas ONNX export
        print("Exporting best model to ONNX...")
        model.load_state_dict(
            torch.load(
                "models/" + args.filename + ".pt", map_location=torch.device("cpu")
            )
        )
        model = model.to("cpu")
        model.eval()
        input_tensor = torch.randn(1, 1, 1, 490)
        bo.export_qonnx(model, input_tensor, "models/" + args.filename + ".onnx")
        print(f"Exported model to models/{args.filename}.onnx")

    else:
        # Testing the model
        total = 0
        correct = 0
        loss = 0.0

        model.load_state_dict(
            torch.load("models/" + args.filename + ".pt", map_location=device)
        )
        model.eval()
        error.eval()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                prediction = model.forward(images)
                loss += error(prediction, labels).item() * images.size(0)

                pred = prediction.data.argmax(1, keepdim=True)
                correct += pred.eq(labels.data.view_as(pred)).sum().item()
                total += images.size(0)

        loss /= total
        accuracy = correct * 100.0 / total
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%")
