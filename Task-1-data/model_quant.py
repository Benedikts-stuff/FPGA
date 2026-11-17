import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import brevitas.nn as qnn
import argparse
import numpy as np
from losses import SqrHingeLoss
import brevitas.export as bo
import brevitas.quant as bq
import os
import matplotlib.pyplot as plt


class KWS(Module):
    def __init__(self, weight_bit_width=4, act_bit_width=4):
        super(KWS, self).__init__()
        # TODO: Your model definition here:
        self.in_quant = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.fc0 = qnn.QuantLinear(490, 256, weight_bit_width=weight_bit_width, bias=False)
        self.bn0 = nn.BatchNorm1d(256)
        self.act0 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.fc1 = qnn.QuantLinear(256, 256, weight_bit_width=weight_bit_width, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.fc2 = qnn.QuantLinear(256, 128, weight_bit_width=weight_bit_width, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.fc3 = qnn.QuantLinear(128, 11, weight_bit_width=weight_bit_width, bias=False)

    def forward(self, x):
        # TODO: Your forward pass here

        # This is how you flatten the input for linear layers
        x = x.view(x.shape[0], -1)
        # ToTensor() transformation made inputs fall in the range [0,1]
        # The following line makes this [-1,1]
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        # Quantize the input
        x = self.in_quant(x)
        # Actual inference
        x = self.act0(self.bn0(self.fc0(x)))
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantized model for keyword spotting")
    parser.add_argument("filename", type=str, nargs="?", default="kws_w4a4", help="Output file name")
    parser.add_argument(
        "--epochs", type=int, nargs="?", default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--gpu", type=str, nargs="?", default=0, help="Index of GPU if available"
    )
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--seed", type=int, nargs="?", default=5, help="Random seed")
    parser.add_argument(
        "--lr", type=float, nargs="?", default=0.005, help="Learning rate"
    )
    parser.add_argument("--bsize", type=int, nargs="?", default=256, help="Batchsize")
    parser.add_argument(
        "--weight_bits", type=int, default=4, help="Weight bit width"
    )
    parser.add_argument(
        "--act_bits", type=int, default=4, help="Activation bit width"
    )

    args = parser.parse_args()
    bsize = args.bsize
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs("models", exist_ok=True)

    # Automatically infer bit-width from filename if encoded, or use explicit arguments
    w_bits = args.weight_bits
    a_bits = args.act_bits
    fn_lower = args.filename.lower() if args.filename else ""
    if "w8a8" in fn_lower:
        w_bits, a_bits = 8, 8
    elif "w4a4" in fn_lower:
        w_bits, a_bits = 4, 4
    elif "w2a2" in fn_lower:
        w_bits, a_bits = 2, 2
    elif "w2a4" in fn_lower:
        w_bits, a_bits = 2, 4
    elif "w1a2" in fn_lower:
        w_bits, a_bits = 1, 2

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
    ]  # , 'silence'

    # class for preprocessed KWS data
    from torchvision import transforms
    from speechcommands_dataset import SpeechCommandsDataset
    from sklearn.utils import class_weight
    from sklearn.model_selection import train_test_split
    import numpy as np

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device} | Configuration: W{w_bits}A{a_bits}")
    model = KWS(weight_bit_width=w_bits, act_bit_width=a_bits).to(device)

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
    data = np.load("data/kws_test_X.npy")
    labels = np.load("data/kws_test_Y.npy")
    test_set = SpeechCommandsDataset(
        data, labels, transform=transforms.Compose([transforms.ToTensor()])
    )

    # encapsulate data into dataloader form
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=bsize, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=bsize, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=bsize, shuffle=False
    )

    # Set your preferred loss function here
    error = torch.nn.CrossEntropyLoss().to(device)

    # Set your preferred optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TODO: Set the learning rate (you can use a scheduler instead)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.train:
        highestAcc = 0.0
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        for epoch in range(args.epochs):
            model.train()
            error.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                # Transfering images and labels to GPU if available
                images, labels = images.to(device), labels.to(device)

                prediction = model.forward(images)

                loss = error(prediction, labels)
                # Initializing a gradient as 0 so there is no mixing of gradient among the batches
                optimizer.zero_grad()
                # Propagating the error backward
                loss.backward()
                # Optimizing the parameters
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

            # Testing the model on validation set
            total = 0
            correct = 0
            val_loss = 0.0

            model.eval()
            error.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    prediction = model.forward(images)
                    loss = error(prediction, labels)

                    val_loss += loss.item() * images.size(0)
                    pred = prediction.data.argmax(1, keepdim=True)
                    correct += pred.eq(labels.data.view_as(pred)).sum().item()
                    total += images.size(0)

            val_loss /= total
            val_accuracy = correct * 100.0 / total
            val_loss_list.append(val_loss)
            val_acc_list.append(val_accuracy)

            if val_accuracy > highestAcc:
                highestAcc = val_accuracy
                torch.save(
                    obj=model.state_dict(), f="models/" + args.filename + ".pt"
                )

            print(
                "Epoch: {}, Loss: {:.4f}, Accuracy: {:.2f}% (Train Acc: {:.2f}%)".format(
                    epoch, val_loss, val_accuracy, train_acc
                )
            )

        # Save accuracy and loss history
        history = {
            "train_loss": train_loss_list,
            "train_acc": train_acc_list,
            "val_loss": val_loss_list,
            "val_acc": val_acc_list,
        }
        np.save("models/" + args.filename + "_history.npy", history)

        # Plot and save visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(range(1, args.epochs + 1), train_acc_list, label="Train Accuracy", marker="o")
        ax1.plot(range(1, args.epochs + 1), val_acc_list, label="Val Accuracy", marker="s")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title(f"Accuracy Curve (W{w_bits}A{a_bits})")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(range(1, args.epochs + 1), train_loss_list, label="Train Loss", marker="o")
        ax2.plot(range(1, args.epochs + 1), val_loss_list, label="Val Loss", marker="s")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title(f"Loss Curve (W{w_bits}A{a_bits})")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("models/" + args.filename + "_accuracy.png", dpi=150)
        plt.close()

        # brevitas onnx export
        model.load_state_dict(
            torch.load(
                "models/" + args.filename + ".pt", map_location=torch.device("cpu")
            )
        )
        model = model.to("cpu")
        model.eval()
        input_tensor = torch.randn(1, 1, 1, 490)
        bo.export_qonnx(model, input_tensor, "models/" + args.filename + ".onnx")
        print("Exported ONNX model to models/" + args.filename + ".onnx")

    else:
        # Testing the model
        total = 0
        correct = 0
        loss = 0.0
        model.load_state_dict(
            torch.load("models/" + args.filename + ".pt", map_location=device)
        )
        model = model.to(device)
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
        print("Loss: {:.4f}, Accuracy: {:.2f}%".format(loss, accuracy))
