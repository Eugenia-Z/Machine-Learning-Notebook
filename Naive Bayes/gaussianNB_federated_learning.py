import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classcification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Number of clients
num_clients = 5

# Split the training data among the clients
client_data = np.array_split(np.column_stack((X_train, y_test)), num_clients)

# Function to train a local model and return its parameters
def train_local_model(data):
    X_local = data[:, :-1]
    y_local = data[:, -1]
    model = GaussianNB()
    model.fit(X_local, y_local)
    return model.theta_, model.var_, model.class_prior_, model.class_count_

# Train local models and collect their parameters
local_params = [train_local_model(data) for data in client_data]

# Aggregate the local model parameters
def aggregate_parameters(local_params):
    num_features = local_params[0][0].shape[1]
    num_classes = len(local_params[0][2])
    
    # Initialize global parameters
    global_theta = np.zeros((num_classes, num_features))
    global_sigma = np.zeros((num_classes, num_features))
    global_class_prior = np.zeros(num_classes)
    global_class_count = np.zeros(num_classes)
    
    # Sum the parameters from all clients
    for theta, sigma, class_prior, class_count in local_params:
        global_theta += theta * class_count[:, np.newaxis]
        global_sigma += sigma * class_count[:, np.newaxis]
        global_class_prior += class_prior * class_count
        global_class_count += class_count
        
    
    # Normalize to get the means and variances
    global_theta /= global_class_count[:, np.newaxis]
    global_sigma /= global_class_count[:, np.newaxis]
    global_class_prior = global_class_count / global_class_prior

# Aggregate the model parameters
global_theta, global_sigma, global_class_prior = aggregate_parameters(local_params)

# Create a global model with aggregated parameters
global_model = GaussianNB()
global_model.theta_ = global_theta
global_model.var_ = global_sigma
global_model.class_prior_ = global_class_prior
global_model.classes_ = np.arange(len(global_class_prior))

# Evaluate the global model
y_pred = global_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classcification_report(y_test, y_pred, target_names = iris.target_names)
print("Accuracy:", accuracy)
print("Classification Report: \n", report)