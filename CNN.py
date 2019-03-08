###Source code from http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/###
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn               # import torch.nn = from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import oned_implicit_heatconduction
import sys
import numpy as np
from torch.autograd import Variable 
import matplotlib.pyplot as plt

torch.manual_seed(0)

#data generation parameters
num_samples = 2000 #number of samples
vary_left_iso_bc = True
vary_diffusivity = False

print("Total number of samples: {}".format(num_samples))
print("Vary left isothermal BC: {}, Vary thermal diffusivity: {}".format(vary_left_iso_bc,vary_diffusivity))

#parameters
num_epochs = 1000000
num_classes = 1
learning_rate = 0.00000001   

print("Number of epochs: {}".format(num_epochs))
print("Number of classes: {}".format(num_classes))
print("Learning rate: {}".format(learning_rate))

test_case = False
net_model = 'MLP'

if test_case == True:
   dataset = np.zeros((N,10,10), dtype=np.float64)
   labels = np.arange(N, dtype = np.float32)
   labels = labels.reshape(len(labels),1)

   dataset[0,0,:] = np.arange(10, dtype = np.float64)

   for i in range(1,10):
      dataset[0, i, :] = dataset[0, i - 1, :] + 10

   for i in range(1,N):
      dataset[i, :, :] = dataset[i - 1, :, :] + 100

   dataset = torch.tensor(dataset.astype(np.float32))
   labels = torch.tensor(labels.astype(np.float32))

   if net_model is 'CNN':
      dataset = dataset.unsqueeze(1)

else:
   dataset, labels = oned_implicit_heatconduction.main(num_samples, vary_left_iso_bc, vary_diffusivity)
   labels = labels[:,0]
   labels = labels.reshape(len(labels),1)
   # convert numpy array to pytorch tensor
   dataset = torch.tensor(dataset.astype(np.float32))
   # labels = labels.reshape(len(labels),1)
   labels = torch.tensor(labels.astype(np.float32))

   dataset = dataset.permute(2,0,1)   #permute columns so 1st dimension is num_samples, 2nd is num_TCs, 3rd is num_tie_steps
   dataset = dataset.unsqueeze(1)  #add a dimension for channels

#divide dataset into training and test
#note we do not need to randomly extract from dataset since there is already randomness due
#to how the data was generated

num_samples, num_input_channels, num_time_steps, num_TCs = dataset.size()

num_training_samples = round(0.75*num_samples)    #3:1 train:test split

training_dataset = dataset[:num_training_samples]
training_labels = labels[:num_training_samples]
test_dataset = dataset[num_training_samples:,:,:,:]
test_labels = labels[num_training_samples:]


#function for computing image width or height after convolving or pooling
def calc_size(orig_size, filter_size, padding, stride, layer):
   length = (orig_size - filter_size + 2*padding)/stride + 1

   if length.is_integer() is False:
      print("Filter size in layer {} resulted in non-integer dimension".format(layer))
      sys.exit()

   return int(length)

class CNN(nn.Module):
   def __init__(self):
      super().__init__()

      filter_size_width_1 = 3
      filter_size_height_1 = filter_size_width_1
      padding_1 = 0
      stride_1 = 1
      num_filters_1 = 16

      filter_size_width_2 = 3
      filter_size_height_2 = filter_size_width_2  
      padding_2 = 0
      stride_2 = 1
      num_filters_2 = 64

      self.layer1 = nn.Sequential(
         nn.Conv2d(num_input_channels, num_filters_1, kernel_size = filter_size_width_1, stride = stride_1, padding = padding_1),      #input_channels = 1 b/c grayscale
         nn.ReLU())#,
         #nn.MaxPool2d(kernel_size = 2, stride = 2))

      #image starts out as num_TCs x num_time_steps (10x100)
      #now we compute the dimensions after 1st conv layer
      W = calc_size(num_TCs, filter_size_width_1, padding_1, stride_1, 1)  #=6
      H = calc_size(num_time_steps, filter_size_height_1, padding_1, stride_1, 1)  #=6

      print("w type", type(W))

      print("Width: {}, height: {} after first convolution".format(W,H))

      self.layer2 = nn.Sequential(
         nn.Conv2d(num_filters_1, num_filters_2, kernel_size = filter_size_width_2, stride = stride_2),
         nn.ReLU())#,
         #nn.MaxPool2d(kernel_size = 2, stride = 2))

      #image starts out as num_TCs x num_time_steps (10x100)
      #now we compute the dimensions after 1st conv layer
      W = calc_size(W, filter_size_width_2, padding_2, stride_2, 2)  
      H = calc_size(H, filter_size_height_2, padding_2, stride_2, 2)

      print("Width: {}, height: {} after first convolution".format(W,H))

      self.drop_out = nn.Dropout()      #no input paramters

      #specify 2 fully connected layers
      self.fc1 = nn.Linear(W * H * num_filters_2, 1000)
      self.fc2 = nn.Linear(1000, num_classes)      

   def forward(self, data_to_propagate):       #analogous to virtual functions in C++, we're overriding the forward method in base class (nn.Module)
      out = self.layer1(data_to_propagate)
      out = self.layer2(out)
      out = out.reshape(out.size(0), -1)       #reshape to 1st order tensor for FC layer
      out = self.drop_out(out)
      out = F.relu(self.fc1(out))
      out = F.relu(self.fc2(out))

      return out

class MLP(nn.Module):

   def __init__(self):
      super().__init__()
      self.num_layers = 3
      num_nodes_per_layer = [200, 100, num_classes]

      self.fc = nn.ModuleList([nn.Linear(num_TCs * num_time_steps, num_nodes_per_layer[0])])
      for i in range(1, self.num_layers):
         self.fc.append(nn.Linear(num_nodes_per_layer[i - 1], num_nodes_per_layer[i]))

      #hyperparameters
      # num_layers_1 = 200
      # num_layers_2 = 100

      # self.fc1 = nn.Linear(num_TCs * num_time_steps, num_layers_1)
      # self.fc2 = nn.Linear(num_layers_1, num_layers_2)
      # self.fc3 = nn.Linear(num_layers_2, num_classes)

   def forward(self, x, num_samples):       #analogous to virtual functions in C++, we're overriding the forward method in base class (nn.Module)
      out = x.reshape(num_samples, num_TCs * num_time_steps)

      for i in range(self.num_layers - 1):      
         out = F.relu(self.fc[i](out))

      i = i + 1    #python doesn't increment i at end of loop
      out = self.fc[i](out)

      # out = F.relu(self.fc1(out))
      # out = F.relu(self.fc2(out))
      # out = self.fc3(out)
      return out

#create CNN instance
if net_model is 'CNN':
   model = CNN()
else:
   model = MLP()

#check if cuda is enabled and transfer dataset to GPU if cuda is enabled
is_cuda = torch.cuda.is_available()

if is_cuda:
   cuda = torch.device('cuda') 
   training_dataset = training_dataset.to(device = cuda)
   training_labels = training_labels.to(device = cuda)
   test_dataset = test_dataset.to(device = cuda)
   model = model.to(device = cuda)

print("CUDA IS:",  is_cuda)
print("Using", net_model, "architecture")

#Loss and optimizer
cost_func = nn.MSELoss()      #this contains both cross entropy and softmax
#weight_decay is the regularization constant
#we can set a momentum value between (0,1) to get past local minima
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)#, weight_decay = 1e-5)

#training stage
loss_list = []
mse_list = []
epoch_list = []

for epoch in range(num_epochs):
   outputs = model(training_dataset, num_training_samples)
   loss = cost_func(outputs, training_labels)

   #backpropagation and optimization
   optimizer.zero_grad()
   loss.backward()             #calculates gradients in back propagation
   optimizer.step()            #after coputing gradients, perform optimization

   #track acuracy
   # predicted = outputs.data
   # predicted = predicted.squeeze(1)   #change it from 2D 1 column array to 1D array
   # mse_err = torch.sum((predicted.float() - training_labels) ** 2)    #output of sum() is a tensor
   # mse_list.append(mse_err)
   loss_list.append(loss)
   epoch_list.append(epoch)
   print("Epoch: {} | MSE: {}".format(epoch, loss))

#plot loss vs epoch
fig = plt.figure()
ax = plt.gca()
ax.plot(epoch_list, loss_list, 'o', c='blue', alpha=0.05, markeredgecolor='none')
ax.set_yscale('log')

#evaluate on test data
predicted = model(test_dataset, num_samples - num_training_samples)
predicted = predicted.cpu()

MSE_test = [None]*num_classes
for i in range(num_classes):
   MSE_test[i] = cost_func(predicted[:,i], test_labels[:,i])

test_labels = test_labels.numpy()
predicted = predicted.cpu().data.numpy()

plt.figure(2)
plt.scatter(test_labels, predicted, s=5)
plt.xlabel('Exact Thermal Conductivity')
plt.ylabel('Predicted Thermal Conductivity')
plt.savefig('predicted_labels_vs_exact.png')
