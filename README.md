# Discrete Time Neural Survival Models Library
## Introduction 
This Python library provides a comprehensive suite of tools for building and evaluating Discrete Time Neural Survival (DTNS) models using PyTorch. Unlike traditional survival analysis that may handle time as a continuous variable, DTNS discretizes time into intervals. This discretization simplifies the modeling process and better aligns with neural network architectures. The library includes modules for data discretization, model building, and loss calculation.

## Features
- Discretization of Event Times: Partition continuous event times into discrete bins for modeling.
- Neural Network Models: Implementations of discrete-time neural network models with customizable layers and activation functions.
- Loss Functions: Specialized loss functions like Negative Log-Likelihood and Cross-Entropy Loss tailored for survival analysis.
- Model Evaluation: Concordance Index to evaluate performance of time to event prediction

## Tutorial 
### Import DTNS library 
```
from DTNS import (
    CI,
    DiscreteFailureTimeCEL,
    DiscreteFailureTimeNLL,
    DiscreteTimeNN,
    create_batches,
    discretize_times,
)
```
### Partition time using `discretize_times`
Assume we have a continuous time tensor and we want to partition the time tensor into four bins where the start and end of each bin are tensors defined below:
```
    times = torch.tensor([0.5, 1.5, 2.5, 3.5])
    bin_starts = torch.tensor([0.0, 1.0, 2.0, 3.0])
    bin_ends = torch.tensor([1.0, 2.0, 3.0, 4.0])
```
We could partion the `times` by directly calling `discretize_times`.
```
print(discretize_times(times, bin_starts, bin_ends))
tensor([[ True, False, False, False],
        [False,  True, False, False],
        [False, False,  True, False],
        [False, False, False,  True]])
```
The result tensor is 4x4 with booleans indicating which bin each continuous time belong. Note that each row represents a time, and each column represents a bin. This can be used separatly, but its primary use is a helper function for later sessions

### Create a model Using `DiscreteTimeNN`
`DiscreteTimeNN` is a class inherented from `nn.module`.
The class has three parameters:
- `hidden_layer_sizes`: A list of integers where each integer denotes the number of neurons in a linear layer. This list determines the structure and depth of the neural network.
- `num_bins`: An integer specifying the number of time intervals.
- `activation`: A PyTorch module class that specifies the activation function to be used after each layer. If not specified, ReLU will be used. This add nonlinearity to the newtwork.  
We could use `DiscreteTimeNN` to create a model
```
    hidden_layer_sizes = [32, 64]
    num_bins = 5
    activation = torch.nn.Tanh
    model = DiscreteTimeNN(hidden_layer_sizes, num_bins, activation)
```
The `model` has two layer, one with 32 neurons and the other has 64 neurons and a Tanh activation function after each layer. There is a final linear layer called `prediction_head `, which has the output size equal to `num_bins + 1`. In the end, the `softmax` function will convert raw output in the `prediction_head` to probabilities. Note that the probabilities refers to the likihood of surviving before previous bins and have event in this bin. We have 1 extra prediction compared to number of bins, and that represents the probability of survive through all the bins and have event beyond time horizon. 

### Calculating Negative Log Likelihood Loss Using `DiscreteFailureTimeNLL`
The Python class `DiscreteFailureTimeNLL` is a module designed to calculate the Negative Log-Likelihood Loss (NLL) for models that predict outcomes over discrete time bins.

$
\text{NLL} = -\sum_{i=1}^n \left[ \delta_i \log(p(T_i)) + (1 - \delta_i) \log(1 - p(T_i)) \right]
$

This class has a hepler function `_get_proportion_of_bins_completed`, which takes survival time as input and outoput a tensor inidcates the proportion of each bin that is completed by each time. This helps to interpolate the event time to smooth transition between bins and allows the model to capture the continuity of the time. In this case, loss will be different for two  censored patients with different observed time, eventhough they have same prediction.
Assume we have following predictions:
```
    bin_boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    tolerance = 1e-8
    loss_fn = DiscreteFailureTimeNLL(bin_boundaries, tolerance)
    predictions = torch.tensor([[0.2, 0.3, 0.4, 0.1, 0.0]])
    event_indicators = torch.tensor([0])
    event_times = torch.tensor([3.5])
    loss = loss_fn(predictions, event_indicators, event_times)
    print(loss)
```
The output is 
```
    tensor(2.9957)
```
In this example, we divide survival time into four bins from one to four. For a patient, we have 5 predictions which represents the predicted probability of event happening in each time interval and beyond. Assume this patient does not have the event and survive up to 3.5(4th bin), the negative log likelihood is $ -log(1 - (0.2 + 0.3 + 0.4 + 0.1 \cdot 0.5)) = 2.995732274$. Result from calculator aligns with the result from our function. The NLL loss for the entire model will be mean of the NLL loss across all the data points. 

### Calculating Cross Entropy Loss Using `DiscreteFailureTimeCEL`
The class `DiscreteFailureTimeCEL` is a PyTorch module designed to calculate the cross entropy loss, which  inherits from `torch.nn.Module`.  
$\text{Cross Entropy Loss} = -\sum \left( y \log(p) + (1 - y) \log(1 - p) \right)$  
Using the same example:
```
    bin_boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    tolerance = 1e-8
    loss_fn = DiscreteFailureTimeCEL(bin_boundaries, tolerance)
    predictions = torch.tensor([[0.2, 0.3, 0.4, 0.1, 0.0]])
    event_times = torch.tensor([3.5])
    loss = loss_fn(predictions, event_times)
    print(loss)
```
The output is
```
tensor(3.3932)
```
Note that we no longer care about the event status in this case.
The cross entropy loss in this case is:   
$
-(0\cdot log(0.2)+(1-0)\cdot log(1-0.2))\\
-(0\cdot log(0.3)+(1-0)\cdot log(1-0.3))\\
-(0\cdot log(0.4)+(1-0)\cdot log(1-0.4))\\
-(1\cdot log(0.1)+(1-1)\cdot log(1-0.1))
$  
Simplify we have:
$
-(log(0.8)+log(0.7)+log(0.6)+log(0.1)) = 3.393229212
$   
This aligns with the calculations from our function. This loss punishes for high probability on the wrong interval and encourages high confidence on the correct interval. 
### Create Batch Using `create_batches`:
`create_batches` is a utility function that could help to create batch from tensors for the neural network. This function has two parameters:
- `*arrs`: Variable length argument list of arrays or tensors.
- `batch_size`: The size of each batch. Must be greater than 0.
The function calculates the length of the input same length arrays, and then
iterates over the indices of the arrays in steps equal to batch_size, creating slices of each array from the current index ndx to ndx + batch_size.
For each iteration, it yields a tuple where each element of the tuple is a batch (slice) converted into a tensor. 
```
x = [1, 2, 3, 4, 5, 6]
y = [7, 8, 9, 10, 11, 12]

for batch in create_batches(x, y, batch_size=2):
    print(batch)
```
The output will be:
```
(tensor([1, 2]), tensor([7, 8]))
(tensor([3, 4]), tensor([9, 10]))
(tensor([5, 6]), tensor([11, 12]))
```
### Calculating Concordance Index Using `CI`:
This is a utility function that calculates the concordance index. The concordance index indicates what is probability a model correctly rank predicted risk for two individuals. The parameter for this function is:
- `s_test` (np.ndarray): An array indicating whether an event has occurred (event indicator). 
- `t_test` (np.ndarray): An array of survival times or times to event/censoring.
- `pred_risk` (np.ndarray): An array of predicted risks, where higher values indicate higher risk of event occurrence.
- `bin` (int): The index of the bin for which the concordance index is calculated. 
- tied_tol (float, default = 1e-8): A tolerance level used to determine when risk differences are considered tied.  
Note that `pred_risk`for individual is the sum of the probability having the event before and at a certain time interval specified by `bin`.
```
    s_test = np.array([1, 0, 1, 0, 1])
    t_test = np.array([5, 8, 3, 4, 2])
    pred_risk = np.random.rand(5, 10)
    bin = 4
    tied_tol = 1e-8
    CI = CI(s_test, t_test, pred_risk, bin, tied_tol)
    print(CI)
```
`pred_risk` here is a 5x10 array of random values representing the predicted risk across 10 time bins for each of the 5 data points. Then the concordance index is calculated on these prediciton at 4th bin using the true survival time and survival status `t_test` and `s_test`
The output is:
```
0.5
```
This implies that the model genereted this `pre_risk` has 50% probability of correctly ranking 2 individual. Since we randomly generated `pre_risk`, this result is expected. 

## For Contributors
Contribution to my library is always welcomed and appreciated.
I have local test that ensures correct values will be returned for all three function, and you can test your code by typing  `python -m pytest tests/` . Feel free to use or edit my test. 

## Acknowledgement
This work is inspired by Dr. Matthew Engelhard's notebook. https://github.com/engelhard-lab/collaborative-ml-notebooks/blob/main/notebooks/discrete_time_neural_survival_model.ipynb   
I adapted `DiscreteTimeNN`, `create_batches`,and `DiscreteFailureTimeNLL` but I made some modification such as adding flexibility and extra checks. I wrote `DiscreteFailureTimeCEL` and `CI` by myself. All the tests and materials in README are also Original.










