#Reference: https://www.youtube.com/watch?v=mozBidd58VQ&list=LL&index=9&t=918s, by Nicholas Renotte
import torch
from PIL import Image
from torch import nn, save, load #neural network class
from torch.optim import AdamW #implements AdamW algorithm
import torch.optim as optim
from torch.utils.data import DataLoader #to load buit-in data set from pytorch
import os
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor #convert an image into tenser which is what will work in pytorch

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learningRate = 1e-3
weight_decay = 1e-5
batchSize = 5



#We need to downlaod our data set
train = datasets.FER2013(root='/Users/MarcoKrause/Desktop/EmotionData', split='train', transform= ToTensor())
dataset = DataLoader(train, 32) #
RESOURCES = {
        "train": ("train.csv", "3f0dfb3d3fd99c811a1299cb947e3131"),
        "test": ("test.csv", "b02c2298636a634e8c2faabbf3ea9a23"),
    }
#1,48,48 is the image shape of our dataset

#Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self, train): #this method is used to define the layers of the network, such as convolutional layers, linear layers, activation functions, etc
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,32,(3,3)), #input=1=black and white, output=32, kernel size of 3*3 to capture finer details in the image
           #torch.nn.Conv2d(in_channels, out_channels, kernel_size)
            nn.ReLU(),
            nn.Conv2d(32,64,(3,3)),
            nn.ReLU(),
            nn.Conv2d(64,64,(3,3)),
            nn.ReLU(),
           #shaving 2 pixels of height and width of the image each time
          nn.Flatten(),
           #after flatten the layers, we will pass the input shape into linear layer...
          nn.Linear(64*(48-6)*(48-6), 7) #in_features= the (number of channels after the shaving process)*(side1-6pixel)*(side2-6pixel)
           #7 classes in total
        )


    def forward(self, x):
        return self.model(x)
print('hello')

#Loss, optimizer
if torch.cuda.is_available():
    model = ImageClassifier(train=True).to('cuda')
else:
    model = ImageClassifier(train=True).to('cpu')

optimizer = optim.AdamW(model.parameters(), lr = learningRate, weight_decay = weight_decay)
lossFunc = nn.CrossEntropyLoss()

#Training
if __name__ == "__main__":
    '''   
    for epoch in range(10): #10 epoches
        for batch in dataset:
            X,y = batch
            X, y = X.to('cpu'), y.to('cpu')
            yhat = model(X)
            loss = lossFunc(yhat, y)

            #Apply backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch:{epoch} loss is {loss.item()}") #not sure what this is 

    with open('model_state.pt', 'wb') as file:
        save(model.state_dict(), file)
    '''
    with open('model_state.pt', 'rb') as file:
        model.load_state_dict(load(file))

    image = Image.open('image9.jpg')
    image_tensor = ToTensor()(image).unsqueeze(0).to('cpu')

    #0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    print(torch.argmax(model(image_tensor))) 
    #torch.argmax() returns the indices of the maximum values of a tensor across a dimension
    

if torch.argmax(model(image_tensor)) == 0:
    emotion = "Angry"
    print(f"Emotion: {emotion}")

elif torch.argmax(model(image_tensor)) == 1:
    emotion = "Disgust"
    print(f"Emotion: {emotion}")

elif torch.argmax(model(image_tensor)) == 2:
    emotion = "Fear"
    print(f"Emotion: {emotion}")
elif torch.argmax(model(image_tensor)) == 3:
    emotion = "Happy"
    print(f"Emotion: {emotion}")
elif torch.argmax(model(image_tensor)) == 4:
    emotion = "Sad"
    print(f"Emotion: {emotion}")
elif torch.argmax(model(image_tensor)) == 5:
    emotion = "Surprise"
    print(f"Emotion: {emotion}")
elif torch.argmax(model(image_tensor)) == 6:
    emotion = "Neutral"
    print(f"Emotion: {emotion}")
'''
    for epoch in range(10): #10 epoches
        for batch in dataset:
            X,y = batch
            X, y = X.to('cuda'), y.to('cuda')
            yhat = model(X)
            loss = lossFunc(yhat, y)

            #Apply backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch:{epoch} loss is {loss.item()}") #not sure what this is 

    with open('model_state.pt', 'wb') as file:
        save(model.state_dict(), file)
'''

import ChatGPT_API_EmotionalSupport
from ChatGPT_API_EmotionalSupport import get_completion
import openai
import os
from dotenv import load_dotenv
prompt = f"The person in front of me is feeling {emotion}, what can I do to help?"
print(prompt)
response = get_completion(prompt)

print(response)

