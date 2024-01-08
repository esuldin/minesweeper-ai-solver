import numpy
import torch.nn
import torch.optim
import torch.utils.data
from minesweeper_ai_solver.model import MinesweeperSolver

from minesweeper import Mode, GameState, CellState
from minesweeper_game.game_field import MinesweeperGame


class MinesweeperFieldVectorizer:
    def __call__(self, field, dtype=numpy.float32):
        v1 = numpy.zeros((11,) + field.shape, dtype=dtype)

        v1[0] = numpy.ones(field.shape)
        v1[1] = (field != CellState.CLOSED)

        cell_states = [CellState.NO_MINES_NEARBY, CellState.ONE_MINE_NEARBY, CellState.TWO_MINES_NEARBY,
                       CellState.THREE_MINES_NEARBY, CellState.FOUR_MINES_NEARBY, CellState.FIVE_MINES_NEARBY,
                       CellState.SIX_MINES_NEARBY, CellState.SEVEN_MINES_NEARBY, CellState.EIGHT_MINES_NEARBY]
        for idx, state in enumerate(cell_states):
            v1[idx + 2] = (field == state)

        v2 = (field == CellState.CLOSED).astype(dtype)

        return v1, v2


class MinesweeperSolverDataSet(torch.utils.data.Dataset):
    def __init__(self, fields, predictions, vectorizer):
        self._fields = fields
        self._predictions = predictions
        self._vectorizer = vectorizer

    def __len__(self):
        return len(self._fields)

    def __getitem__(self, idx):
        return self._vectorizer(self._fields[idx]), self._predictions[idx]


class MinesweeperSolverTrainer:
    def __init__(self, game_mode):
        self._game_mode = game_mode
        self._model = MinesweeperSolver()
        self._optimizer = torch.optim.Adam(self._model.parameters())
        #self._optimizer = torch.optim.SGD(self._model.parameters(), lr=0.01)
        self._loss_fn = torch.nn.BCELoss()

    def train(self, batches, batch_size):
        torch.autograd.set_detect_anomaly(True)
        field_vectorizer = MinesweeperFieldVectorizer()

        for batch_idx in range(batches):
            games_played = 0
            games_won = 0
            cells_revealed = 0
            batch_field_states = []
            batch_predictions = []

            samples_taken = 0

            self._model.eval()
            with torch.no_grad():
                while samples_taken < batch_size:
                    games_played += 1
                    game = MinesweeperGame(self._game_mode)
                    game.open((game.mode().height() + 1) * game.mode().width() // 2)

                    while game.state() == GameState.IN_PROGRESS and samples_taken < batch_size:
                        batch_field_states.append(numpy.copy(game.field()))

                        model_input = torch.torch.utils.data.default_collate([field_vectorizer(game.field())])
                        prediction = torch.squeeze(self._model(model_input[0]).detach())
                        prediction = torch.mul(prediction, torch.from_numpy(game.field() == CellState.CLOSED))

                        shifted_prediction = prediction + torch.from_numpy(game.field() != CellState.CLOSED)
                        cell_idx = shifted_prediction.argmin().item()
                        prediction[cell_idx // game.mode().width(), cell_idx % game.mode().width()] = \
                            1 if game.open(cell_idx) == GameState.GAME_OVER else 0

                        batch_predictions.append(prediction)
                        samples_taken += 1

                    cells_revealed += numpy.sum(game.field() != CellState.CLOSED)
                    if game.state() == GameState.WIN:
                        games_won += 1

            running_loss = 0.

            training_set = MinesweeperSolverDataSet(batch_field_states, batch_predictions, field_vectorizer)
            training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size)

            self._model.train()
            for i, data in enumerate(training_loader):
                # Every data instance is an input + label pair
                field_vector_desc, decision = data

                # Zero your gradients for every batch!
                self._optimizer.zero_grad()

                # Make predictions for this batch
                predictions = self._model(field_vector_desc[0])

                # Compute the loss and its gradients
                loss = self._loss_fn(predictions, decision)
                loss.backward()

                # Adjust learning weights
                self._optimizer.step()

                # Gather data and report
                running_loss += loss.item()

                print('batch {} loss: {}'.format(i + 1, running_loss))
                running_loss = 0.

                print('games played: {} games won: {} cells revealed per game: {}'.format(games_played, games_won, cells_revealed/games_played))
                #print(self._model._conv1.weight)
            self._model.eval()
