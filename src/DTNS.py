"""A library that facilitate Discrete Time Neural Survival(DTNS) models.

The functions
"""

from typing import Any, Iterator, List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn


# helper function
def discretize_times(
    times: torch.Tensor, bin_starts: torch.Tensor, bin_ends: torch.Tensor
) -> torch.Tensor:
    """Discretizes the given event times based on the bin boundaries.

    Parameters:
        times (torch.Tensor): The event times to discretize.
        bin_starts (torch.Tensor): The start points of each bin.
        bin_ends (torch.Tensor): The end points of each bin.

    Returns:
        torch.Tensor: A binary tensor indicating whether each time falls
        into each bin.
    """
    return (times[:, None] > bin_starts) & (times[:, None] <= bin_ends)


class DiscreteTimeNN(nn.Module):
    """A discrete time neural network model implementing encoder layers.

    The encoder layer an optional activation function and a softmax function.

    Parameters:

        hidden_layer_sizes(list):
            indcates number of linear layers and neuronsin the model.

        num_bins(int):
            indicates number of bins that time is partitioned into.

        activation(nn.Module):
            Activation function. Without specification, the
            model will use ReLu.

    Attribute:

        layers (nn.ModuleList):
            A list of layers comprising the encoder layers
            with batch normalization and activation, followed by the prediction
            head.

        prediction_head (nn.LazyLinear):
            A linear layer with lazy initialization that serves as the
            prediction head of the model.

        activation(nn.Module):
            Activation function. Without specification, the
            model will use ReLu.

        softmax(torch.nn.Softmax):
            Softmax classifier that converts logit to probability.

    Methods:
        forward(x: torch.Tensor):
            torch.Tensor: Defines the forward pass of the model. The input
            tensor `x` is passed sequentially through the layers in
            `self.layers`, followed by the `prediction_head`, and finally
            through a softmax function to produce a probability distribution
            over the output bins.
    """

    def __init__(
        self,
        hidden_layer_sizes: List[int],
        num_bins: int,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """Initializing the parameters for the model."""
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        for size in hidden_layer_sizes:
            torch.nn.LazyLinear(size)
        self.activation = activation()
        self.prediction_head: nn.LazyLinear = nn.LazyLinear(num_bins + 1)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass.

        A forward pass with customized hidden layer size and activation
        function.
        """
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.prediction_head(x)
        x = self.softmax(x)
        return x


class DiscreteFailureTimeNLL(torch.nn.Module):
    """A PyTorch module for calculating the Negative Log-Likelihood Loss.

    Attributes:
        bin_starts (torch.Tensor): The start points of each bin.
        bin_ends (torch.Tensor): The end points of each bin.
        bin_lengths (torch.Tensor): The lengths of the bins.
        tolerance (float): A small value added for numerical stability
    """

    def __init__(self, bin_boundaries: torch.Tensor, tolerance: float = 1e-8):
        """Initializes parameters for DiscreteFailureTimeNLL."""
        super().__init__()
        if not isinstance(bin_boundaries, torch.Tensor):
            bin_boundaries = torch.tensor(bin_boundaries, dtype=torch.float32)

        self.bin_starts = bin_boundaries[:-1]
        self.bin_ends = bin_boundaries[1:]
        self.bin_lengths = self.bin_ends - self.bin_starts
        self.tolerance = tolerance

    def _get_proportion_of_bins_completed(
        self, times: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the proportion of each bin that is completed.

        Parameters:
            times (torch.Tensor): The event times to evaluate.

        Returns:
            torch.Tensor: The proportion of each bin that is completed by each
            time.
        """
        proportions = (times[:, None] - self.bin_starts) / self.bin_lengths
        return torch.clamp(proportions, min=0, max=1)

    def forward(
        self,
        predictions: torch.Tensor,
        event_indicators: torch.Tensor,
        event_times: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the negative log-likelihood loss.

        Parameters:
            predictions (torch.Tensor):
                The predicted probabilities for each bin.

            event_indicators (torch.Tensor):
                Binary indicators of whether an event occurred or was
                censored.(1 stands for occurrance and 0 for censored)

            event_times (torch.Tensor):
                The times at which events occurred or were censored.

        Returns:
            torch.Tensor: The mean negative log-likelihood of the given data.
        """
        event_likelihood = (
            torch.sum(
                discretize_times(event_times, self.bin_starts, self.bin_ends)
                * predictions[:, :-1],
                dim=1,
            )
            + self.tolerance
        )
        nonevent_likelihood = (
            1
            - torch.sum(
                self._get_proportion_of_bins_completed(event_times)
                * predictions[:, :-1],
                dim=1,
            )
            + self.tolerance
        )

        log_likelihood = event_indicators * torch.log(event_likelihood) + (
            1 - event_indicators
        ) * torch.log(nonevent_likelihood)
        return -torch.mean(log_likelihood)


class DiscreteFailureTimeCEL(torch.nn.Module):
    """A PyTorch module for calculating the Negative Log-Likelihood Loss.

    Attributes:
        bin_starts (torch.Tensor): The start points of each bin.
        bin_ends (torch.Tensor): The end points of each bin.
        bin_lengths (torch.Tensor): The lengths of the bins.
        tolerance (float): A small value added for numerical stability
    """

    def __init__(self, bin_boundaries: torch.Tensor, tolerance: float = 1e-8):
        """Initializes parameters for DiscreteFailureTimeNLL."""
        super().__init__()
        if not isinstance(bin_boundaries, torch.Tensor):
            bin_boundaries = torch.tensor(bin_boundaries, dtype=torch.float32)

        self.bin_starts = bin_boundaries[:-1]
        self.bin_ends = bin_boundaries[1:]
        self.bin_lengths = self.bin_ends - self.bin_starts
        self.tolerance = tolerance

    def forward(
        self,
        predictions: torch.Tensor,
        event_times: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the cross entropy loss.

        Parameters:
            predictions (torch.Tensor):
                The predicted probabilities for each bin.

            event_times (torch.Tensor):
                The times at which events occurred or were censored.

        Returns:
            torch.Tensor: The mean negative log-likelihood of the given data.
        """
        true_distributions = discretize_times(
            event_times, self.bin_starts, self.bin_ends
        ).float()
        cross_entropy_loss = -torch.sum(
            true_distributions
            * torch.log(predictions[:, :-1] + self.tolerance)
            + (1 - true_distributions)
            * torch.log(1 - predictions[:, :-1] + self.tolerance),
            dim=1,
        )
        return torch.mean(cross_entropy_loss)


# utility function
def create_batches(
    *arrs: Any, batch_size: int = 1
) -> Iterator[Tuple[torch.Tensor, ...]]:
    """Generates batches of data from multiple arrays or tensors.

    Parameters:
        *arrs: Variable-length argument list of arrays or tensors.
        batch_size (int): The size of each batch. Must be greater than 0.

    Yields:
        An iterator over tuples of tensors, where each tuple corresponds to a
        batch from the input arrays or tensors.

    Raises:
        ValueError: If `batch_size` is not positive or if input arrays do not
        have the same length.
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive. Got {batch_size}.")

    if not all(len(arr) == len(arrs[0]) for arr in arrs):
        raise ValueError("Input arrays must have the same length.")

    length = len(arrs[0])
    for ndx in range(0, length, batch_size):
        yield tuple(
            torch.as_tensor(arr[ndx : ndx + batch_size]) for arr in arrs
        )


def CI(
    s_test: np.ndarray,
    t_test: np.ndarray,
    pred_risk: np.ndarray,
    bin: int,
    tied_tol: float = 1e-8,
) -> float:
    """A function that calculates Concordance Index at a certain bin.

    Parameters:
        s_test (np.ndarray): An array indicating whether an event
        has occurred (event indicator).

        t_test (np.ndarray): An array of survival times or
        times to event/censoring.

        pred_risk (np.ndarray): An array of predicted risks.

        bin(int): The index of the bin for which the
        concordance index is calculated.

        tied_tol (float, default = 1e-8): A tolerance level used to determine
        when risk differences are considered tied.

    Yields:
        A float indicating the Concordance index in certain bin.
    """
    mask1 = s_test == 1

    mask2 = np.ones_like(s_test, dtype=bool)

    valid = t_test[mask1, np.newaxis] < t_test[np.newaxis, mask2]

    culmulative_risk = np.cumsum(pred_risk, axis=1)[:, :-1]

    culmulative_risk = culmulative_risk[:, bin]

    risk_diff = (
        culmulative_risk[mask1, np.newaxis]
        - culmulative_risk[np.newaxis, mask2]
    )

    correctly_ranked = valid & (risk_diff > tied_tol)
    tied = valid & (np.abs(risk_diff) <= tied_tol)

    num_valid = np.sum(valid)
    if num_valid == 0:
        return 0.0

    ci = np.sum(correctly_ranked + 0.5 * tied) / num_valid

    return float(ci)
