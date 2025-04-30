import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Path to the directory containing your TensorBoard logs
route_log="logs"

# Create an EventAccumulator
event_acc = event_accumulator.EventAccumulator(route_log)

# Reload the data from the logs
event_acc.Reload()


# obtain all tags from tensors
all_tensors=event_acc.Tags()['']
print(all_tensors)

# Print the data
for tag in all_tensors:
    #I only write the loss
    if tag == "epoch_loss":
        loss_data = event_acc.Tensors(tag)
        for event in loss_data:
            tensor_data = event.tensor_proto
            tensor_content = np.frombuffer(tensor_data.tensor_content, dtype=np.float32)
            print(f"Tag: {tag}, Step: {event.step}, Tensor_Data:{tensor_content[0]},{event.value}")