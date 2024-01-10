import boto3
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from fathomnet.api import images, boundingboxes, taxa
from tqdm import tqdm
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ffmpeg
import glob
import random
import numpy as np
import openai
import os
import re
import requests
import torch
import traceback
import wandb
import boto3

BUCKET_NAME = 'seabot-d2-storage'
S3_MODEL_ROOT_PATH = "SeaBot/Models"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

wandb.login(key='856878a46a17646e66281426d43c4b77d3f9a00c')
wandb.init(project="seabot", name=f"fn_finetuning_vit_pretrained")

def download_from_s3(bucket_name, s3_path, local_path):
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(bucket_name, s3_path, local_path)
        print(f"Successfully downloaded {s3_path} from S3 bucket {bucket_name} to {local_path}")
    except Exception as e:
        print(f"Error occurred while downloading from S3: {e}")


def save_model_to_s3(local_model_path, s3_model_path, bucket_name):
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(local_model_path, bucket_name, s3_model_path)
        print(f"Model successfully uploaded to {s3_model_path} in bucket {bucket_name}")
    except Exception as e:
        print(f"Error occurred while uploading model to S3: {e}")


# Define the custom dataset class for handling FathomNet data
class FathomNetDataset(Dataset):
    def __init__(self, fathomnet_root_path, concepts, transform=None):
        self.transform = transform
        self.images_info = []
        self.image_dir = fathomnet_root_path
        self.concepts = concepts
        self.concept_to_index = {concept: i for i, concept in enumerate(concepts)}

        print("Number of classes in set: " + str(len(concepts)))

        # Fetch image data for each concept and save the information
        for concept in concepts:
            try:
                images_info = images.find_by_concept(concept)
                self.images_info.extend(images_info)
            except ValueError as ve:
                print(f"Error fetching image data for concept {concept}: {ve}")
                continue

        # Sort images info to ensure consistent order across different runs
        self.images_info.sort(key=lambda x: x.uuid)

        # Create directory if it doesn't exist
        os.makedirs(self.image_dir, exist_ok=True)

        # Download images for each image info and save it to disk
        for image_info in tqdm(self.images_info, desc="Downloading images", unit="image"):
          image_url = image_info.url
          image_path = os.path.join(self.image_dir, f"{image_info.uuid}.jpg")

          # Download only if image doesn't already exist
          if not os.path.exists(image_path):
              try:
                  image_data = requests.get(image_url).content
                  with open(image_path, 'wb') as handler:
                      handler.write(image_data)
              except ValueError as ve:
                  print(f"Error downloading image from {image_url}: {ve}")
                  continue

    # Get the number of images in the dataset
    def __len__(self):
        return len(self.images_info)

    # Fetch an image and its label vector by index
    def __getitem__(self, idx):
      try:
          image_info = self.images_info[idx]
          image_path = os.path.join(self.image_dir, f"{image_info.uuid}.jpg")
          image = Image.open(image_path).convert('RGB')

          # Create label vector
          labels_vector = torch.zeros(len(self.concepts))
          for box in image_info.boundingBoxes:
            if box.concept in self.concept_to_index:
              labels_vector[self.concept_to_index[box.concept]] = 1

          # Apply transformations if any
          if self.transform:
            image = self.transform(image)

          return image, labels_vector
      except (IOError, OSError):
          print(f"Error reading image {image_path}. Skipping.")
          return None, None

def collate_fn(batch):
    # Filter out the (None, None) entries from the batch
    batch = [(image, label) for image, label in batch if image is not None and label is not None]

    # If there are no valid items left, return (None, None)
    if len(batch) == 0:
        return None, None

    # Extract and stack the images and labels
    images = torch.stack([item[0] for item in batch])
    labels_vector = torch.stack([item[1] for item in batch])

    return images, labels_vector

def load_and_train_model(model_root_path, old_model_path, fathomnet_root_path):
    # Define a transformation that resizes images to 224x224 pixels and then converts them to tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Find the concepts for the bounding boxes
    concepts = boundingboxes.find_concepts()

    # Create a dataset with the given concepts and the defined transform
    dataset = FathomNetDataset(fathomnet_root_path, concepts, transform=transform)

    # Calculate the sizes for the training and validation datasets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Set a seed for the random number generator
    torch.manual_seed(0)

    # Split the dataset into training and validation subsets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders for the training and validation datasets with batch size of 16
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

    # Load the pre-trained Vision Transformer model and replace the classifier with a new one with 4 classes
    model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
    model.classifier = nn.Linear(model.config.hidden_size, 4)

    # Unfreeze all layers for training
    for param in model.parameters():
        param.requires_grad = True

    # Load the pre-trained model parameters for further training
    model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')

    # Download and load the pre-trained model parameters for further training
    if old_model_path:
        local_model_path = os.path.join(model_root_path, 'best_model_vit.pth')
        download_from_s3(BUCKET_NAME, old_model_path, local_model_path)

        if os.path.isfile(local_model_path):
            model.load_state_dict(torch.load(local_model_path))
            print(f"Loaded the pre-trained model from {local_model_path} for further training.")
        else:
            print(f"Pre-trained model file {local_model_path} not found. Using default weights.")

    # Replace the classifier again, this time with the number of concept classes
    model.classifier = nn.Linear(model.config.hidden_size, len(concepts))

    # Move the model to the GPU if available
    model = model.to(device)

    # Define the optimizer as Adam
    optimizer = optim.Adam(model.parameters())

    # Define the number of training epochs and the patience for early stopping
    num_epochs = 1
    patience = 2
    no_improve_epoch = 0

    # Frequency for saving the model
    save_freq = 100000

    # Replace the StepLR with OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=num_epochs)

    # Define a folder to store checkpoints
    checkpoint_folder = os.path.join(model_root_path, 'fn_checkpoints')

    # Make sure the checkpoint folder exists
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Load the latest checkpoint if it exists
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_folder, "*.pth")))

    # Define a function to load the latest checkpoint
    def load_latest_checkpoint():
        checkpoints = glob.glob(os.path.join(checkpoint_folder, "*.pth"))
        checkpoints.sort(key=lambda x: [int(num) for num in re.findall(r'\d+', x)], reverse=True) # Sorting based on epoch and batch number

        if checkpoints:
            latest_checkpoint_path = checkpoints[0]
            checkpoint = torch.load(latest_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['batch']
            best_loss = checkpoint['best_loss']
            print(f"Loaded Checkpoint from {latest_checkpoint_path}!!")
            return start_epoch, start_batch, best_loss
        else:
            print("No Checkpoint found!!")
            return 0, 0, np.inf


    # Define the loss function as binary cross-entropy with logits
    criterion = nn.BCEWithLogitsLoss()

    # Ensure the model is in the correct device
    model.to(device)

    # Define a function for the training loop
    def train_loop(start_epoch, start_batch, best_loss):
        total_batches = len(train_loader)  # Total number of batches in one epoch
        for epoch in range(start_epoch, num_epochs):
            print(f'Starting epoch {epoch + 1}/{num_epochs}')
            running_loss = 0.0
            model.train()

            for batch_idx, (images, labels_vector) in enumerate(train_loader, start=start_batch):
                if images is None or labels_vector is None:
                    print("Terminating batch due to image or label vector read error.")
                    break

                images = images.to(device)
                labels_vector = labels_vector.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs.logits, labels_vector)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                wandb.log({"fn_epoch": epoch, "fn_loss": loss.item()})

                # Print epoch progress
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Epoch {epoch + 1} Progress: {progress:.2f}%")

                if (batch_idx + 1) % save_freq == 0:
                    checkpoint_path = os.path.join(checkpoint_folder, f'fn_checkpoint_{epoch + 1}_{batch_idx + 1}.pth')
                    torch.save({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        'best_loss': best_loss,
                    }, checkpoint_path)
                    print(f'Saved model checkpoint at {checkpoint_path}')

            epoch_loss = running_loss / len(train_loader.dataset)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                print(f'New best loss: {best_loss}')
                no_improve_epoch = 0  # Reset patience
            
                # Save the best model to S3
                best_model_path = os.path.join(checkpoint_folder, 'fn_best_model_pretrained.pth')
                torch.save(model.state_dict(), best_model_path)
                s3_model_path = os.path.join(S3_MODEL_ROOT_PATH, 'fn_best_model_pretrained.pth')
                save_model_to_s3(best_model_path, s3_model_path, BUCKET_NAME)
            else:
                no_improve_epoch += 1

            if no_improve_epoch >= patience:
                print(f'Early stopping after {patience} epochs without improvement.')
                break

    # Load the latest checkpoint and start/resume training
    start_epoch, start_batch, best_loss = load_latest_checkpoint()
    train_loop(start_epoch, start_batch, best_loss)

    final_model_path = os.path.join(checkpoint_folder, 'fn_final_trained_model_pretrained.pth')
    torch.save(model.state_dict(), final_model_path)

    # Save the final model to S3
    s3_final_model_path = os.path.join(S3_MODEL_ROOT_PATH, 'fn_final_trained_model_pretrained.pth')
    save_model_to_s3(final_model_path, s3_final_model_path, BUCKET_NAME)


model_root_path = "local_models"
old_model_path = "SeaBot/Models/best_model_vit.pth"

# Current working directory
current_working_dir = os.getcwd()

# Relative path for 'fathomnet' directory within the current working directory
fathomnet_relative_path = "fathomnet"

# Joining the paths
fathomnet_root_path = os.path.join(current_working_dir, fathomnet_relative_path)

final_model_path = os.path.join(current_working_dir, 'fn_trained_model_pretrained.pth')

load_and_train_model(model_root_path, old_model_path, fathomnet_root_path)
