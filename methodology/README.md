# Methodology

Code, notebooks, and results for the expected-goals (xG) model in Rocket League. The central question: **does encoding team structure with a spectral embedding give a better xG model than raw positional (real-unit) features?**

Pipeline: parsed replay CSVs → preprocessing (features + sequences) → LSTM training/comparison → statistical analysis → xG prediction and shot analysis.

## Files

### Preprocessing

| File | Description |
|---|---|
| `preprocessing.py` | Original script. Converts raw replay CSVs into 64-frame sequences, computes features, and saves a torch `TensorDataset`. |
| `new_preprocessing.py` | Revised version of the above. Moves `sign()` out of the triangle function and updates the column indices used to detect the player count. |
| `experiment_preprocessing.py` | Version used for the model-comparison experiment. Builds both the real-unit and spectral datasets and adds `keep_first_half()` to trim each segment from 64 to 32 frames. |
| `experiment_preprocessing.ipynb` | Step-by-step walkthrough of the experiment preprocessing, with the reasoning behind each feature. |

**Features computed per time-step**

- Ball position and speed
- 3D distance from the ball to the center of the goal
- Angle subtended by the goal posts at the ball
- Number of defenders inside the ball–goal triangle (with $z \le 646$)
- Number of players within 400 units of the ball
- **Spectral model:** a 1D team-state per team, plus each team's distance to the ball state
- **Real-unit model:** xyz centroid for each team, plus each centroid's distance to the ball

**Spectral embedding.** For each frame, build a graph over the ball and players and compute similarities with an RBF kernel,

$$S(i,j) = \exp\!\left(-\frac{d(i,j)^2}{2\sigma^2}\right).$$

Players on opposite teams get zero similarity. Form the graph Laplacian $L = D - S$ and take the Fiedler (bottom non-trivial) eigenvector as a 1D representation. Each team's cluster value is the mean of its players' entries.

### Model

| File | Description |
|---|---|
| `model.py` | Defines `xG_LSTM` (stacked unidirectional LSTM followed by a small MLP head, outputting a shot-outcome logit) and `initialize_model()`, which loads the saved weights from `models/`. `predict_xG()` applies a sigmoid to return a probability. |
| `experiment_model_training.ipynb` | Trains the real-unit and spectral LSTMs over 20 random seeds each. Logs train/validation/test loss, AUC, epochs to convergence, and aggregated feature importances. |
| `predict_xG.ipynb` | Runs the trained model over preprocessed replay data to produce xG values. |

### Results and analysis

| File | Description |
|---|---|
| `experiment_training_results.csv` | Output of the training experiment: one row per seed, with metrics and feature importances for both models. Read by the analysis notebook. |
| `experiment_results_analysis.ipynb` | Compares the two models metric by metric using raincloud/ridge plots, a two-sample Cramér–von Mises test (same distribution?), and one-sided Mann–Whitney U tests (which model is better?). |
| `research_questions.ipynb` | Follow-up analyses of the model's xG output: what separates high-xG shots ($\text{xG} \ge 0.65$) from low-xG shots ($\text{xG} \le 0.35$), how high-xG non-goals compare to goals, and how well the game-state variables encode team and ball states. |
| `graph_plotting.ipynb` | Field and graph visualizations: draws the pitch, plots player/ball positions, and renders the per-frame similarity graph. |

### Utilities

| File | Description |
|---|---|
| `csv_test.py` | Scratch script for checking the shape and columns of raw, formatted, and preprocessed CSVs. |
