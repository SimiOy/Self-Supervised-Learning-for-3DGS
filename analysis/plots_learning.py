import re
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 22
})

def parse_log_data(log_entries):
    # Define regex patterns to extract data
    epoch_start_pattern = re.compile(r"Epoch (\d+)")
    total_loss_pattern = re.compile(r"Total Loss: ([\d\.]+), CV Loss: ([\d\.]+), CM Loss: ([\d\.]+)")
    cm_accuracy_pattern = re.compile(r"CM Accuracy: ([\d\.]+)")
    distance_pattern = re.compile(r"Positive Pairs: ([\d\.]+), Negative Pairs: ([\d\.]+)")

    # Structures to hold parsed data
    epoch_losses = []
    cv_losses = []
    cm_losses = []
    epoch_cm_accuracy = []
    epoch_distances = []
    current_epoch = 0

    for line in log_entries.split('\n'):
        if 'Total Loss' in line:
            total_loss_match = total_loss_pattern.search(line)
            if total_loss_match:
                total_loss = float(total_loss_match.group(1))
                cv_loss = float(total_loss_match.group(2))
                cm_loss = float(total_loss_match.group(3))
                epoch_losses.append((current_epoch, total_loss))
                cv_losses.append((current_epoch, cv_loss))
                cm_losses.append((current_epoch, cm_loss))

        elif 'CM Accuracy' in line:
            cm_accuracy_match = cm_accuracy_pattern.search(line)
            if cm_accuracy_match:
                cm_accuracy = float(cm_accuracy_match.group(1))
                epoch_cm_accuracy.append((current_epoch, cm_accuracy))

        elif 'CV Mean Pair Distance' in line:
            distance_match = distance_pattern.search(line)
            if distance_match:
                positive_pairs = float(distance_match.group(1))
                negative_pairs = float(distance_match.group(2))
                distance_diff = abs(positive_pairs - negative_pairs)
                epoch_distances.append((current_epoch, distance_diff))

        elif 'Epoch' in line:
            epoch_match = epoch_start_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

    return epoch_losses, cv_losses, cm_losses, epoch_cm_accuracy, epoch_distances


def plot_metrics(epoch_losses, cv_losses, cm_losses, epoch_cm_accuracy, epoch_distances):
    # Extract data for plotting
    epochs, _ = zip(*epoch_losses)
    _, cv_values = zip(*cv_losses)
    _, cm_values = zip(*cm_losses)
    _, accuracies = zip(*epoch_cm_accuracy) if epoch_cm_accuracy else ([], [])
    epoch_nums, distance_diffs = zip(*epoch_distances) if epoch_distances else ([], [])

    # Create plots with a custom grid layout
    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)  # Loss plots span two columns
    ax2 = plt.subplot2grid((2, 2), (1, 0))  # CM Accuracy plot
    ax3 = plt.subplot2grid((2, 2), (1, 1))  # Distance Differences plot

    # Plotting CV and CM Losses
    index = list(range(len(cv_losses)))
    ax1.plot(index, cv_values, label='CV Loss')
    ax1.plot(index, cm_values, label='CM Loss', alpha=0.7)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss Values')
    ax1.set_title('CV and CM Losses per Epoch')
    ax1.grid()
    ax1.legend()

    # Plotting CM Accuracy
    if accuracies:
        ax2.plot(_, accuracies, label='CM Accuracy')
        ax2.set_xlabel('Epochs')
        # ax2.set_ylabel('CM pretext task')
        ax2.set_title('CM Accuracy per Epoch')
        ax2.grid()
        ax2.legend()

    # Plotting Distance Differences
    if epoch_distances:
        ax3.plot(epoch_nums, distance_diffs, label='mPD Difference')
        ax3.set_xlabel('Epochs')
        # ax3.set_ylabel('CV pretext task')
        ax3.set_title('CV mPD Differences per Epoch')
        ax3.grid()
        ax3.legend()

    # Customizing x-axis to show epoch numbers only every 5 epochs
    epoch_starts = [i for i, e in enumerate(epochs) if i == 0 or epochs[i] != epochs[i - 1]]
    filtered_starts = [start for start in epoch_starts if epochs[start] % 5 == 0]
    filtered_labels = [epochs[start] for start in filtered_starts]

    ax1.set_xticks(filtered_starts)
    ax1.set_xticklabels(filtered_labels)

    # Show the plots
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # load log file
    model_name = 'pointnet_uniform_sampling'
    file_path = f'../log/{model_name}/logs/pointnet_cls.txt'
    with open(file_path, 'r') as file:
        log_content = file.read()

    losses, cv_losses, cm_losses, accuracies, distances = parse_log_data(log_content)
    plot_metrics(losses, cv_losses, cm_losses, accuracies, distances)
