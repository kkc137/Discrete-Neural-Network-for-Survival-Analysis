"""Test DNTS."""

import math
import unittest

import numpy as np
import torch
from DTNS import (
    CI,
    DiscreteFailureTimeCEL,
    DiscreteFailureTimeNLL,
    DiscreteTimeNN,
    create_batches,
    discretize_times,
)


def test_discretize_times_within_boundaries() -> None:
    """Test case: Event times falling within the bin boundaries.

    This test function checks if the discretize_times function correctly
    discretizes event times that fall within the given bin boundaries. It
    creates sample event times and bin boundaries, calls the discretize_times
    function, and asserts that the resulting binary tensor matches the
    expected output.

    Raises:
        AssertionError: If the discretize_times function produces unexpected
        results for event times within the bin boundaries.
    """
    times = torch.tensor([0.5, 1.5, 2.5, 3.5])
    bin_starts = torch.tensor([0.0, 1.0, 2.0, 3.0])
    bin_ends = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected_output = torch.tensor(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ]
    )
    assert torch.all(
        discretize_times(times, bin_starts, bin_ends) == expected_output
    )


def test_discretize_times_outside_boundaries() -> None:
    """Test case: Event times falling outside the bin boundaries.

    This test function checks if the discretize_times function correctly
    handles event times that fall outside the given bin boundaries. It creates
    sample event times and bin boundaries, calls the discretize_times function,
    and asserts that the resulting binary tensor matches the expected output.

    Raises:
        AssertionError: If the discretize_times function produces unexpected
        results for event times outside the bin boundaries.
    """
    times = torch.tensor([-0.5, 4.5])
    bin_starts = torch.tensor([0.0, 1.0, 2.0, 3.0])
    bin_ends = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected_output = torch.tensor(
        [[False, False, False, False], [False, False, False, False]]
    )
    assert torch.all(
        discretize_times(times, bin_starts, bin_ends) == expected_output
    )


def test_discretize_times_edge_cases() -> None:
    """Test case: Edge cases where event times coincide with bin boundaries.

    This test function checks if the discretize_times function correctly
    handles edge cases where event times coincide with the bin boundaries. It
    creates sample event times and bin boundaries, calls the discretize_times
    function, and asserts that the resulting binary tensor matches the
    expected output.

    Raises:
        AssertionError: If the discretize_times function produces unexpected
        results for event times that coincide with bin boundaries.
    """
    times = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    bin_starts = torch.tensor([0.0, 1.0, 2.0, 3.0])
    bin_ends = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected_output = torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ]
    )
    assert torch.all(
        discretize_times(times, bin_starts, bin_ends) == expected_output
    )


class TestDiscreteTimeNN(unittest.TestCase):
    """Test the forward pass of the DiscreteTimeNN model."""

    def test_forward_pass(self) -> None:
        """Test the forward pass of the DiscreteTimeNN model.

        This test case checks if the forward pass of the DiscreteTimeNN model
        produces the expected output shape and if the output is a valid
        probability distribution. It creates an instance of the model with
        specified hidden layer sizes, number of bins, and activation function,
        and performs the forward pass with a randomly generated input tensor.

        The test case asserts that the output tensor has the expected shape
        (batch_size, num_bins + 1) and that the sum of probabilities along the
        last dimension is close to 1.0, indicating a valid probability
        distribution.

        Raises:
            AssertionError: If the output tensor does not have the expected
            shape or if the output is not a valid probability distribution.
        """
        # Define the test input
        batch_size = 2
        input_size = 10
        x = torch.randn(batch_size, input_size)

        # Define the model parameters
        hidden_layer_sizes = [32, 64]
        num_bins = 5
        activation = torch.nn.Tanh

        # Create an instance of the DiscreteTimeNN model
        model = DiscreteTimeNN(hidden_layer_sizes, num_bins, activation)

        # Perform the forward pass
        output = model(x)

        # Check the output shape
        expected_output_shape = (batch_size, num_bins + 1)
        self.assertEqual(output.shape, expected_output_shape)

        # Check if the output is a valid probability distribution
        self.assertTrue(torch.allclose(output.sum(dim=1), torch.tensor(1.0)))


class TestDiscreteFailureTimeNLL(unittest.TestCase):
    """Test DiscreteFailureTimeNLL."""

    def setUp(self) -> None:
        """Set up for testing."""
        self.bin_boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        self.tolerance = 1e-8
        self.loss_fn = DiscreteFailureTimeNLL(
            self.bin_boundaries, self.tolerance
        )

    def test_forward_pass_with_event(self) -> None:
        """Test the forward pass when an event occurs.

        This test case checks if the forward pass of the DiscreteFailureTimeNLL
        loss function correctly calculates the negative log-likelihood loss
        when an event occurs within the given bins.
        """
        predictions = torch.tensor([[0.2, 0.3, 0.4, 0.1, 0.0]])
        event_indicators = torch.tensor([1])
        event_times = torch.tensor([2.5])

        loss = self.loss_fn(predictions, event_indicators, event_times)
        likelihood = 0.4
        expected_loss = -math.log(likelihood)

        self.assertAlmostEqual(loss.item(), expected_loss, places=4)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

    def test_forward_pass_with_censoring(self) -> None:
        """Test the forward pass when an event is censored.

        This test case checks if the forward pass of the DiscreteFailureTimeNLL
        loss function correctly calculates the negative log-likelihood loss
        when an event is censored within the given bins.
        """
        predictions = torch.tensor([[0.2, 0.3, 0.4, 0.1, 0.0]])
        event_indicators = torch.tensor([0])
        event_times = torch.tensor([3.5])

        loss = self.loss_fn(predictions, event_indicators, event_times)
        # Manual calculation of the expected loss
        likelihood = 1 - (0.2 + 0.3 + 0.4 + 0.1 * 0.5)
        expected_loss = -math.log(likelihood)

        self.assertAlmostEqual(loss.item(), expected_loss, places=4)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

    def test_forward_pass_batch(self) -> None:
        """Test the forward pass with a batch.

        This test case checks if the forward pass of the DiscreteFailureTimeNLL
        loss function correctly calculates the mean negative log-likelihood
        loss when given a batch of predictions, event indicators, and event
        times.
        """
        predictions = torch.tensor(
            [[0.2, 0.3, 0.4, 0.1, 0.0], [0.1, 0.2, 0.3, 0.3, 0.1]]
        )
        event_indicators = torch.tensor([1, 0])
        event_times = torch.tensor([2.5, 3.5])

        loss = self.loss_fn(predictions, event_indicators, event_times)

        likelihood_event = 0.4
        expected_loss_event = -math.log(likelihood_event)

        likelihood_censor = 1 - (0.1 + 0.2 + 0.3 + 0.3 * 0.5)
        expected_loss_censor = -math.log(likelihood_censor)

        expected_loss = (expected_loss_event + expected_loss_censor) / 2
        self.assertAlmostEqual(loss.item(), expected_loss, places=4)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)


class TestDiscreteFailureTimeCEL(unittest.TestCase):
    """A unittest.TestCase class to test the DiscreteFailureTimeCEL module.

    This test class verifies the correct implementation
    of the DiscreteFailureTimeCEL module,
    which calculates cross-entropy loss for discrete-time
    survival analysis. The tests
    ensure that the loss computed by the module
    aligns with the expected loss derived
    from manual calculations. The cross-entropy
    loss is a measure of the difference
    between two probability distributions –
    the actual distribution of event times and
    the predicted distribution by the model.

    Attributes:
        bin_boundaries (torch.Tensor):
        Boundaries used for discretizing continuous event times.
        tolerance (float): A small constant to prevent
        division by zero in logarithmic calculations.
        loss_fn (DiscreteFailureTimeCEL): An instance of
        the loss module with set bin boundaries and tolerance.

    Methods:
        setUp: Initializes the test conditions with
        bin boundaries, tolerance, and loss function instance.
        test_forward_pass: Checks the forward calculation
        of the loss module against manually computed values.
    """

    def setUp(self) -> None:
        """Set up the test case.

        This method initializes the bin boundaries, tolerance, and creates an
        instance of the DiscreteFailureTimeCEL module.
        """
        self.bin_boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        self.tolerance = 1e-8
        self.loss_fn = DiscreteFailureTimeCEL(
            self.bin_boundaries, self.tolerance
        )

    def test_forward_pass(self) -> None:
        """Test the forward pass of the DiscreteFailureTimeCEL module.

        This test case checks if the forward pass of the DiscreteFailureTimeCEL
        module correctly calculates the cross-entropy loss for given prediction
        and event times.

        The test creates sample predictions and event times, calls the forward
        pass of the module, and compares the resulting loss with a manually
        calculated expected loss.

        The expected loss is calculated by discretizing the event times,
        computing the cross-entropy loss for each sample, and taking the mean.

        The test uses the `assertAlmostEqual` method to compare the loss values
        allowing for a small tolerance (4 decimal places) in the comparison.

        Raises:
            AssertionError: If the calculated loss does not match the expected
                loss within the specified tolerance.
        """
        predictions = torch.tensor(
            [[0.2, 0.3, 0.4, 0.1, 0.0], [0.1, 0.2, 0.3, 0.3, 0.1]]
        )
        event_times = torch.tensor([2.5, 3.5])

        loss = self.loss_fn(predictions, event_times)

        # Manual calculation of the expected loss
        true_distributions = discretize_times(
            event_times, self.bin_boundaries[:-1], self.bin_boundaries[1:]
        ).float()
        cross_entropy_loss = -torch.sum(
            true_distributions
            * torch.log(predictions[:, :-1] + self.tolerance)
            + (1 - true_distributions)
            * torch.log(1 - predictions[:, :-1] + self.tolerance),
            dim=1,
        )
        expected_loss = torch.mean(cross_entropy_loss)

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4)


class TestCreateBatches(unittest.TestCase):
    """Test case for the DiscreteFailureTimeCEL module.

    This unit test class checks the functionality of the DiscreteFailureTimeCEL
    module, which implements a custom loss function for discrete-time survival
    analysis. The test ensures that the forward pass of the module calculates
    thecorrect cross-entropy loss for given predictions and event times.

    Attributes:
        bin_boundaries (torch.Tensor): The tensor representing
        the boundaries of the bins for discretization.
        tolerance (float): A small value to ensure
        numerical stability in log operations.
        loss_fn (DiscreteFailureTimeCEL): An instance of the
        DiscreteFailureTimeCEL
        module initialized with the bin boundaries and tolerance.

    Methods:
        setUp: Prepares the necessary components for the test,
        such as bin boundaries and loss function instance.
        test_forward_pass: Validates the correctness of the
        loss computation by comparing it against a manually
        calculated expected loss.
    """

    def setUp(self) -> None:
        """Set up for test."""
        self.arr1 = [1, 2, 3, 4, 5]
        self.arr2 = [6, 7, 8, 9, 10]
        self.tensor1 = torch.tensor(self.arr1)
        self.tensor2 = torch.tensor(self.arr2)

    def test_create_batches_with_example(self) -> None:
        """Check if batches are created as expected."""
        batches = tuple(create_batches(self.arr1, self.arr2, batch_size=2))
        batches_expected = (
            (torch.tensor([1, 2]), torch.tensor([6, 7])),
            (torch.tensor([3, 4]), torch.tensor([8, 9])),
            (torch.tensor([5]), torch.tensor([10])),
        )

        self.assertEqual(len(batches), len(batches_expected))

        for batch, expected in zip(batches, batches_expected):
            self.assertTrue(
                torch.equal(batch[0], expected[0]), "X batches do not match"
            )
            self.assertTrue(
                torch.equal(batch[1], expected[1]), "Y batches do not match"
            )

    def test_batches_of_correct_size(self) -> None:
        """Check if batches are of correct size."""
        batch_size = 2
        batches = list(
            create_batches(self.arr1, self.arr2, batch_size=batch_size)
        )
        for batch in batches:
            self.assertTrue(
                all(
                    tensor.size(0) == batch_size or tensor.size(0) < batch_size
                    for tensor in batch
                )
            )

    def test_raise_value_error_for_non_positive_batch_size(self) -> None:
        """Check if ValueError is raised for non-positive batch size."""
        with self.assertRaises(ValueError):
            list(create_batches(self.arr1, self.arr2, batch_size=0))

    def test_raise_value_error_for_unequal_lengths(self) -> None:
        """Test if ValueError is raised for input arrays of unequal length."""
        with self.assertRaises(ValueError):
            list(create_batches(self.arr1, self.arr2 + [11], batch_size=1))

    def test_yield_batches_as_tuples_of_tensors(self) -> None:
        """Test the batch generator for yielding batches as tuples of tensors."""  # noqa: E501
        batch_size = 1
        batches = list(
            create_batches(self.arr1, self.arr2, batch_size=batch_size)
        )
        for batch in batches:
            self.assertIsInstance(batch, tuple)
            self.assertTrue(
                all(isinstance(tensor, torch.Tensor) for tensor in batch)
            )


class TestCI(unittest.TestCase):
    """Test the Concordance Index Calculation."""

    def setUp(self) -> None:
        """Prepare sample data."""
        self.s_test = np.array([1, 0, 1, 0, 1])
        self.t_test = np.array([5, 8, 3, 4, 2])
        self.pred_risk = np.random.rand(5, 10)
        self.bin = 4
        self.tied_tol = 1e-8

    def test_CI_calculation(self) -> None:
        """Test that CI returns a float and is within expected range."""
        ci_value = CI(
            self.s_test, self.t_test, self.pred_risk, self.bin, self.tied_tol
        )
        self.assertIsInstance(ci_value, float)
        self.assertGreaterEqual(ci_value, 0.0)
        self.assertLessEqual(ci_value, 1.0)

    def test_no_events(self) -> None:
        """Test that CI handles cases with no valid comparisons."""
        s_test_no_events = np.zeros_like(self.s_test)
        ci_value = CI(
            s_test_no_events,
            self.t_test,
            self.pred_risk,
            self.bin,
            self.tied_tol,
        )
        self.assertEqual(ci_value, 0)
