import csv
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from tqdm import tqdm

from src.dataset.korean_food_dataset import KoreanFoodInferenceDataset
from src.config.cfg_project1 import get_cfg_project1_default


cfg = get_cfg_project1_default()


def initialize_model(model_name, num_classes, checkpoint_path):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(checkpoint_path))
        input_size = 224
    
    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model, input_size

def main():
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "./data"
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"
    # Number of classes in the dataset
    num_classes = 150
    # Batch size for training (change depending on how much memory you have)
    batch_size = 8
    # Initialize the model for this run
    model, input_size = initialize_model(model_name, num_classes, checkpoint_path=cfg.CHECKPOINT_PATH)
    # Print the model we just instantiated
    print(model)
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    test_datasets = KoreanFoodInferenceDataset(cfg.TEST_DATA_PATH, data_transforms['test'])
    # Create training and validation dataloaders
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=4)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model = model.to(device)
    # Set model as inference model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    # Inference
    result = []
    for images, image_ids in tqdm(test_dataloader):
        num_image = images.shape[0]
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for i in range(num_image):
            result.append({
                'image_id': image_ids[i],
                'class': preds[i].item()
            })
    # Save to csv
    with open('submission.csv', 'w') as f:
        submission_writer = csv.writer(f)
        submission_writer.writerow(['filename','class'])
        for res in result:
            submission_writer.writerow([res['image_id'], res['class']])

if __name__ == '__main__':
    main()