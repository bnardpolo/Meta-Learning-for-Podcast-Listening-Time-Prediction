
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

class ImprovedPrototypicalNetwork:
    """
    Global regressor (Ridge) + per-task adaptation using support-set residual.
    Adds robust preprocessing:
      - converts inf/-inf -> NaN
      - SimpleImputer(median) to fill NaNs
      - StandardScaler for stable training
    """
    def __init__(self, alpha=1.0, adapt_lr=0.5, clip_min=None, clip_max=None, random_state=42):
        self.alpha = float(alpha)
        self.adapt_lr = float(adapt_lr)
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.global_model = Ridge(alpha=self.alpha, random_state=random_state)

        self._is_fit = False

    def _clean(self, A):
        A = np.asarray(A, dtype=float)
        # turn +/-inf into NaN (the imputer will handle them)
        A[~np.isfinite(A)] = np.nan
        return A

    def fit(self, X, y):
        X = self._clean(X)
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        self.global_model.fit(X, np.asarray(y, dtype=float))
        self._is_fit = True
        return self

    def predict(self, task, k_shot=5):
        """
        task: dict with keys:
            - 'X': 2D array-like features (support + query)
            - 'y': 1D array-like targets (same length as X rows)
            - 'n_episodes': int total episodes (len(y))
        k_shot: number of support examples from the start of the task
        """
        if not self._is_fit:
            raise RuntimeError("ImprovedPrototypicalNetwork must be fit() before predict().")

        X = self._clean(task["X"])
        y = np.asarray(task["y"], dtype=float)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D X, got shape {X.shape}")

        if len(y) != X.shape[0]:
            raise ValueError("Length of y must match number of rows in X for the task.")

        if k_shot < 1 or k_shot >= len(y):
            raise ValueError(f"k_shot must be in [1, {len(y)-1}] for this task.")

        support_X = X[:k_shot]
        support_y = y[:k_shot]
        query_X   = X[k_shot:]

        # Same preprocessing path at inference
        support_X = self.imputer.transform(support_X)
        query_X   = self.imputer.transform(query_X)
        support_X = self.scaler.transform(support_X)
        query_X   = self.scaler.transform(query_X)

        # Global prediction
        global_pred = self.global_model.predict(query_X)

        # Simple adaptation: bias by mean residual on the support set
        support_pred = self.global_model.predict(support_X)
        resid = (support_y - support_pred).mean() if support_y.size else 0.0
        adapted = global_pred + self.adapt_lr * resid

        # Optional clipping in the model (you can also clip in evaluation)
        if self.clip_min is not None or self.clip_max is not None:
            adapted = np.clip(adapted,
                              self.clip_min if self.clip_min is not None else -np.inf,
                              self.clip_max if self.clip_max is not None else  np.inf)
        return adapted
