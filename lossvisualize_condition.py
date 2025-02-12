import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = '/hdd_ext/hdd3/joowoniese/diffwave4/conditionalDiffwave/event_logs/loss_log.csv'
data = pd.read_csv(csv_file)

# Print data to check structure
print(data)

# Check if 'grad_norm' values are in the expected format and extract the numeric part
def extract_grad_norm(x):
    try:
        # Attempt to extract the numeric value from 'tensor()'
        return float(x.split('(')[1].split(',')[0])
    except IndexError:
        # If the format is not as expected, return NaN or a default value
        return None

data['grad_norm'] = data['grad_norm'].apply(extract_grad_norm)
data['loss'] = data['loss'].astype(float)  # Ensure loss is float

# Plotting loss graph
plt.figure(figsize=(10, 6))
plt.plot(data['step'], data['loss'], color='tab:blue', label='Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss Graph')
plt.grid(True)
plt.legend()
plt.savefig("/hdd_ext/hdd3/joowoniese/diffwave4/conditionalDiffwave/train_visualize/loss_graph.png")
plt.show()

# Plotting grad_norm graph
plt.figure(figsize=(10, 6))
plt.plot(data['step'], data['grad_norm'], color='tab:red', label='Grad Norm')
plt.xlabel('Step')
plt.ylabel('Grad Norm')
plt.title('Grad Norm Graph')
plt.grid(True)
plt.legend()
plt.savefig("/hdd_ext/hdd3/joowoniese/diffwave4/conditionalDiffwave/train_visualize/grad_graph.png")
plt.show()
