import numpy

from torch.utils.data import default_collate

from minesweeper_game.game_interface import CellState

from .model import MinesweeperSolverModel
from .vectorizer import MinesweeperFieldVectorizer


class MinesweeperSolver:
    """
    This is a wrapper for the model to work with one field.
    """
    def __init__(self, model=None, vectorizer=None):
        self._model = model if model is not None else MinesweeperSolverModel()
        self._vectorizer = vectorizer if vectorizer is not None else MinesweeperFieldVectorizer()

        self._model.eval()

    def __call__(self, field):
        # The model always accepts batches, therefore, it is necessary to create a batch with a single element to get
        # a prediction.
        model_input = self._vectorizer(field)
        model_input = default_collate([model_input])

        # The model also returns the prediction as a batch with a single element, therefore, it necessary to change
        # the representation of it.
        model_output = self._model(model_input)
        model_output = model_output.view(field.shape)

        return self._cell_idx(field, model_output), model_output

    def model(self):
        return self._model

    def vectorizer(self):
        return self._vectorizer

    def _cell_idx(self, field, model_prediction):
        next_cell_prediction = None
        next_cell_index = None
        with numpy.nditer(model_prediction.detach().numpy(), ['multi_index']) as predictions_it:
            for value in predictions_it:
                if field[predictions_it.multi_index] == CellState.CLOSED:
                    if next_cell_prediction is None or value < next_cell_prediction:
                        next_cell_prediction = value
                        next_cell_index = predictions_it.iterindex

        return next_cell_index
