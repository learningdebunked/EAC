import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

def _train_neural(self, user_item_matrix: csr_matrix):
    """Train neural collaborative filtering (Apple MPS if available)."""
    # ---- device (MPS if available) ----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    self.logger.info(f"Training neural model on device: {device}")

    # ---- data -> dense -> tensor on device ----
    X_dense = user_item_matrix.toarray()  # be careful: can be huge
    X_tensor = torch.tensor(X_dense, dtype=torch.float32, device=device)

    # ---- model + training setup on device ----
    self.model = self.model.to(device)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    n_users, n_items = X_tensor.shape

    # Build full (user,item) index grid once (on device)
    user_idx = torch.arange(n_users, device=device)
    item_idx = torch.arange(n_items, device=device)
    uu, ii = torch.meshgrid(user_idx, item_idx, indexing="ij")  # shapes: (n_users, n_items)

    # Flatten into paired indices (same length)
    uu_flat = uu.reshape(-1)                 # (n_users*n_items,)
    ii_flat = ii.reshape(-1)                 # (n_users*n_items,)
    target_flat = X_tensor.reshape(-1)       # (n_users*n_items,)

    for epoch in range(50):
        self.model.train()

        # Forward: model should return (batch,) or (batch,1)
        preds_flat = self.model(uu_flat, ii_flat).reshape(-1)

        loss = criterion(preds_flat, target_flat)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            self.logger.info(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

    self.model.eval()