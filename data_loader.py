from ccbdl.data.utils.get_loader import get_loader
import matplotlib.pyplot as plt
import random
from torchvision import transforms
from functools import partial

def transform_label(label, selected_indices):
    return select_labels(label, selected_indices)

def prepare_data(data_config):
    # augmentations_list = get_augmentation(data_config['augmentation'])
    # final_transforms = transforms.Compose(augmentations_list)

    # Retrieve label indices from data_config
    label_indices = data_config.get('label_indices', [])
    # data_config["transform_input"] = final_transforms
    
    # Use partial to pre-fill label_indices in transform_label
    data_config["transform_label"] = partial(transform_label, selected_indices=label_indices)

    loader = get_loader(data_config["dataset"])
    train_data, test_data, validation_data = loader(**data_config).get_dataloader()
    
    #view_data(train_data, data_config)
    #view_data(test_data, data_config)

    return train_data, test_data, validation_data

def get_augmentation(augmentations):
    transform_list = []
    for item in augmentations:
        if isinstance(item, str):  # Direct transform like RandomHorizontalFlip
            transform = getattr(transforms, item)()
            transform_list.append(transform)
        elif isinstance(item, dict):  # Transform with parameters like RandomRotation and Resize
            for name, params in item.items():
                if isinstance(params, list):  # If parameters are given as a list
                    transform = getattr(transforms, name)(*params)
                else:  # If a single parameter is given
                    transform = getattr(transforms, name)(params)
                transform_list.append(transform)
    return transform_list

def select_labels(label, selected_indices):
    # Selects specific labels based on provided indices
    return label[selected_indices]

def view_data(data, data_config):
    # View the first image in train_data or test_data
    label_names = ["Attractive", "Black_hair", "Heavy_Makeup", "High_Cheekbones", "Male", 
                   "No_Beard", "Smiling", "Wearing_Earrings", "Wearing_Lipstick", "Young"]
    batch = next(iter(data))
    inputs, labels = batch
    
    # Set up the subplot dimensions
    fig, axs = plt.subplots(2, 5, figsize=(15, 7))
    axs = axs.ravel()
    
    print(labels.shape)

    for i in range(10):
        image = inputs[i]
        label = labels[i]
        
        # Find labels that are set to 1 and choose one randomly to display
        active_labels = [label_names[j] for j in range(len(label)) if label[j] == 1]
        if active_labels:
            title_label = random.choice(active_labels)
        else:
            title_label = "No Active Label"
            
        image_np = image.permute(1, 2, 0).numpy()
        axs[i].imshow(image_np)
        axs[i].set_title(title_label)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
    
