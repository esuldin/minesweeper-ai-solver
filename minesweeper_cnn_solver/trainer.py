import numpy
import torch.nn
import torch.optim
import torch.utils.data

from minesweeper import Mode, GameState, CellState
from minesweeper_game.game_field import MinesweeperGame

from .solver import MinesweeperSolver


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
    def __init__(self, game_mode, solver):
        self._game_mode = game_mode
        self._solver = solver
        self._optimizer = torch.optim.Adam(self._solver.model().parameters())
        self._loss_fn = torch.nn.BCELoss()

    def train(self, trainer_loop_passes, epochs, batches, batch_size):
        samples_in_epoch = batches * batch_size

        print('Training Iter Idx, Games Played, Games Won, Cells Revealed, Loss')
        for trainer_loop_pass_idx in range(trainer_loop_passes):
            games_played = 0
            games_won = 0
            cells_revealed = 0
            batch_field_states = []
            batch_predictions = []

            samples_taken = 0

            self._solver.model().eval()
            with torch.no_grad():
                while samples_taken < samples_in_epoch:
                    games_played += 1
                    game = MinesweeperGame(self._game_mode)

                    first_cell_idx = numpy.ravel_multi_index((game.mode().height() // 2, game.mode().width() // 2),
                                                             (game.mode().height(), game.mode().width()))
                    game.open(first_cell_idx)

                    while game.state() == GameState.IN_PROGRESS and samples_taken < samples_in_epoch:
                        batch_field_states.append(numpy.copy(game.field()))
                        cell_idx, prediction = self._solver(game.field())

                        sample = prediction.detach()
                        sample[numpy.unravel_index(cell_idx, sample.shape)] = \
                            1 if game.open(cell_idx) == GameState.GAME_OVER else 0
                        batch_predictions.append(sample)

                        samples_taken += 1

                    cells_revealed += numpy.sum(game.field() != CellState.CLOSED)
                    if game.state() == GameState.WIN:
                        games_won += 1

            training_set = MinesweeperSolverDataSet(batch_field_states, batch_predictions, self._solver.vectorizer())
            training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size)

            running_loss = 0.

            self._solver.model().train()
            for epoch_idx in range(epochs):
                for batch_idx, data in enumerate(training_loader):
                    # Every data instance is a field + expected prediction
                    field, expected_prediction = data

                    # Zero your gradients for every batch!
                    self._optimizer.zero_grad()

                    # Make predictions for this batch
                    model_prediction = self._solver.model()(field)

                    # Compute the loss and its gradients
                    loss = self._loss_fn(model_prediction, expected_prediction)
                    running_loss += loss.item() * field.size(0)
                    loss.backward()

                    # Adjust learning weights
                    self._optimizer.step()

            running_loss /= epochs * batches * batch_size
            print('{}, {}, {}, {}, {}'.format(trainer_loop_pass_idx, games_played, games_won, cells_revealed,
                                              running_loss))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train CNN model.')
    parser.add_argument('-g', '--game-mode', help='The Minesweeper game mode to train the model.',
                        default='classic', choices=['classic', 'easy', 'medium', 'expert'])
    parser.add_argument('-t', '--training-iterations', help='The number of training iterations.',
                        default=2000, type=int)
    parser.add_argument('-e', '--epochs', help='The number of epochs during one training iteration.',
                        default=2, type=int)
    parser.add_argument('-b', '--batches', help='The number of batches in one epoch.',
                        default=5, type=int)
    parser.add_argument('-s', '--batch-size', help='The number of samples in one batch.',
                        default=200, type=int)
    parser.add_argument('-o', '--output', help='The path to keep trained model.',
                        default='minesweeper_model.pt')

    args = parser.parse_args()

    if args.game_mode == 'classic':
        selected_game_mode = Mode.CLASSIC
    elif args.game_mode == 'easy':
        selected_game_mode = Mode.EASY
    elif args.game_mode == 'medium':
        selected_game_mode = Mode.MEDIUM
    elif args.game_mode == 'expert':
        selected_game_mode = Mode.EXPERT
    else:
        selected_game_mode = None

    solver = MinesweeperSolver()
    trainer = MinesweeperSolverTrainer(selected_game_mode, solver)
    trainer.train(args.training_iterations, args.epochs, args.batches, args.batch_size)
    solver.model().save(args.output)
