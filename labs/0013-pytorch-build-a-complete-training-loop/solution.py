# --- cell 1 ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):

    history = []

    # 1. Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = X_train.shape[0]

    for epoch in range(epochs):

        # 2. Training mode
        model.train()

        # Shuffle training data
        indices = torch.randperm(n)

        total_train_loss = 0.0

        # 3. Mini-batch training
        for start in range(0, n, batch_size):

            batch_indices = indices[start:start + batch_size]

            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]

            # Clear old gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch_X)

            # Calculate loss
            loss = criterion(logits, batch_y)

            # Backpropagation
            loss.backward()

            # Update parameters
            optimizer.step()

            # Accumulate loss weighted by batch size
            total_train_loss += loss.item() * batch_X.shape[0]

        avg_train_loss = total_train_loss / n

        # 4. Validation
        model.eval()

        with torch.no_grad():

            val_logits = model(X_val)

            val_loss = criterion(val_logits, y_val).item()

            predictions = val_logits.argmax(dim=1)

            val_accuracy = (predictions == y_val).float().mean().item()

        # 5. Store epoch metrics
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })

    return history