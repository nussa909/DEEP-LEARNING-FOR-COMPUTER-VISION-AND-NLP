# %%
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
# filter warnings
warnings.filterwarnings('ignore')

# %%
"""
# Multiclass classification
"""

# %%
"""
## EDA
"""

# %%
df = pd.read_csv('../data/Module_2_Lecture_2_Class_penguins.csv')

# %%
df.sample(5, random_state=42)

# %%
df.info()

# %%
df = df.dropna().reset_index(drop=True)

# %%
plt.figure(figsize=(4,3))
ax = sns.countplot(data=df, x='species')
for i in ax.containers:
    ax.bar_label(i)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
            
plt.suptitle("Target feature distribution")

plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(4,3))
ax = sns.countplot(data=df, x='island')
for i in ax.containers:
    ax.bar_label(i)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
            
plt.suptitle("Island feature distribution")

plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(6,6))
sns.pairplot(data=df, hue='species').fig.suptitle('Numeric features distribution', y=1)
plt.show()

# %%
"""
## Feature preprocessing
"""

# %%
features = ['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

# %%
df = df.loc[:, features]

# %%
df.loc[df['species']=='Adelie', 'species']=0
df.loc[df['species']=='Gentoo', 'species']=1
df.loc[df['species']=='Chinstrap', 'species']=2
df = df.apply(pd.to_numeric)

# %%
df.head(2)

# %%
# Train/test split

X = df.drop('species', axis =1).values
y = df['species'].values

# %%
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %%
X

# %%
X_train , X_test , y_train , y_test = train_test_split(X, y, random_state = 42, test_size =0.33, stratify=y)

# %%
"""
## Modeling
"""

# %%
X_train = torch.Tensor(X_train).float()
y_train = torch.Tensor(y_train).long()

X_test = torch.Tensor(X_test).float()
y_test = torch.Tensor(y_test).long()

# %%
print(X_train[:1])
print(y_train[:10])

# %%
class LinearModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=20, out_dim=3):
        super().__init__()
        
        self.features = torch.nn.Sequential(
            
            nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax()
        )    
        
    def forward(self, x):
        output = self.features(x)
        return output

# %%
model = LinearModel(X_train.shape[1], 20, 3)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epoch = 400 

train_loss = []
test_loss = []

train_accs = []
test_accs = []

for epoch in range(num_epoch):
    
    # train the model
    model.train()
    
    outputs = model(X_train)
    
    loss = criterion(outputs, y_train)    
    train_loss.append(loss.cpu().detach().numpy())
    
    optimizer.zero_grad()    
    loss.backward()
    optimizer.step()
    
    acc = 100 * torch.sum(y_train==torch.max(outputs.data, 1)[1]).double() / len(y_train)
    train_accs.append(acc)
    
    if (epoch+1) % 10 == 0:
        print ('Epoch [%d/%d] Loss: %.4f   Acc: %.4f' 
                       %(epoch+1, num_epoch, loss.item(), acc.item()))
        
    # test the model
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        
        loss = criterion(outputs, y_test)
        test_loss.append(loss.cpu().detach().numpy())
        
        acc = 100 * torch.sum(y_test==torch.max(outputs.data, 1)[1]).double() / len(y_test)
        test_accs.append(acc)

# %%
plt.figure(figsize=(4, 3))
plt.plot(train_loss, label='Train')
plt.plot(test_loss, label='Validation')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training vs Validation Loss')
plt.show()

# %%
plt.figure(figsize=(4, 3))
plt.plot(train_accs, label='Train')
plt.plot(test_accs, label='Validation')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Metric')
plt.show()

# %%
"""
# Regression
"""

# %%
"""
## Pre-processing
"""

# %%
data_path = '../data/Module_2_Lecture_2_Class_bigmart_data.csv'

# %%
data = pd.read_csv(data_path)

# %%
data.head(2)

# %%
# Recreating preprocessing from the ML course

data['Outlet_Establishment_Year'] = 2013 - data['Outlet_Establishment_Year']
data['Item_Visibility'] = (data['Item_Visibility']
                           .mask(data['Item_Visibility'].eq(0), np.nan))

data['Item_Visibility_Avg'] = (data
                               .groupby(['Item_Type',
                                         'Outlet_Type'])['Item_Visibility']
                               .transform('mean'))

data['Item_Visibility'] = (
    data['Item_Visibility'].fillna(data['Item_Visibility_Avg']))

data['Item_Visibility_Ratio'] = (
    data['Item_Visibility'] / data['Item_Visibility_Avg'])

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
    'low fat': 'Low Fat',
    'LF': 'Low Fat',
    'reg': 'Regular'})

data['Item_Identifier_Type'] = data['Item_Identifier'].str[:2]

# %%
data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')

# %%
# train/test split

X_train_num, X_test_num, X_train_cat,  X_test_cat, y_train, y_test = (
    train_test_split(
        data_num.drop(['Item_Outlet_Sales',
                       'Item_Visibility_Avg'], axis=1).values,
        data_cat.drop('Item_Identifier', axis=1).values,
        data['Item_Outlet_Sales'].values,
        test_size=0.2,
        random_state=42))

# %%
num_imputer = SimpleImputer().set_output(transform='pandas')

X_train_num = num_imputer.fit_transform(X_train_num)
X_test_num = num_imputer.transform(X_test_num)

# %%
cat_imputer = SimpleImputer(
    strategy='most_frequent').set_output(transform='pandas')

X_train_cat = cat_imputer.fit_transform(X_train_cat)
X_test_cat = cat_imputer.transform(X_test_cat)

# %%
enc_auto = TargetEncoder(random_state=42).set_output(transform='pandas')

X_train_cat = enc_auto.fit_transform(X_train_cat, y_train)
X_test_cat = enc_auto.transform(X_test_cat)

# %%
X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)

# %%
"""
## Modeling
"""

# %%
# Making a PyTorch Dataset

class BigmartDataset(Dataset):
    def __init__(self, X, y, scale=True):        
        self.X = X.values # from Pandas DataFrame to NumPy array
        self.y = y
        
        if scale:
            sc = StandardScaler()
            self.X = sc.fit_transform(self.X)

    def __len__(self):
        #return size of a dataset
        return len(self.y)

    def __getitem__(self, idx):
        #supports indexing using dataset[i] to get the i-th row in a dataset
        
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)        
        
        return X, y

# %%
# Creating train and test datasets

train_dataset = BigmartDataset(X_train, y_train)
test_dataset = BigmartDataset(X_test, y_test)

# Loading Batches of Data

train_dataloader = DataLoader(train_dataset,
                              batch_size=200,
                              num_workers=4
                             )

test_dataloader = DataLoader(test_dataset,
                              batch_size=200,
                              num_workers=4
                             )

# %%
next(iter(train_dataloader))

# %%
class LinearModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        
        self.features = torch.nn.Sequential(
            nn.Linear(in_dim, 256),
            torch.nn.ReLU(),
            
            nn.Linear(256, 128),
            torch.nn.ReLU(),
            
            nn.Linear(128, 64),
            torch.nn.ReLU(),
            
            nn.Linear(64, out_dim),
        )
    
        
    def forward(self, x):
        output = self.features(x)
        return output

# %%
# Initialize the model
model = LinearModel(in_dim=X_train.shape[1], out_dim=1)
  
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_losses = []
train_rmses = []
test_losses = []
test_rmses = []

# Train the model

num_epochs = 100

for epoch in range(num_epochs):
    
    # Train step
    
    model.train()
    
    y_pred_train = []
    
    for data in train_dataloader:
        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_pred_train.extend(outputs.cpu().detach().numpy())
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, RMSE: {RMSE(y_train, y_pred_train)}')
    train_rmses.append(RMSE(y_train, y_pred_train))
    train_losses.append(loss.cpu().detach().numpy())
    
    # Eval step
    
    model.eval()
    
    y_pred_test = []
    
    with torch.no_grad():
        
        for data in test_dataloader:
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # No backward pass
            
            y_pred_test.extend(outputs.cpu().detach().numpy())
        
        test_rmses.append(RMSE(y_test, y_pred_test))
        test_losses.append(loss.cpu().detach().numpy())
            

# %%
plt.figure(figsize=(4, 3))
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Validation')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Training vs Validation Loss')
plt.show()

# %%
plt.figure(figsize=(4, 3))
plt.plot(train_rmses, label='Train')
plt.plot(test_rmses, label='Validation')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('Training vs Validation Metric - RMSE')
plt.show()