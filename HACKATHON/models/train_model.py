"""
train_model.py
---------------
Yeh script hamare ANN (predictive engine) ko train karti hai.

Kyunke hamare paas labeled real-world "candidate hired ya nahi" dataset
available nahi hai, hum rule-based synthetic tabular dataset generate
karte hain (yeh ek common aur legitimate technique hai jab labeled data
missing ho). Rule: agar skill match, experience, education aur similarity
zyada hain to candidate "Fit" (label=1), warna "Not Fit" (label=0).
Thora random noise bhi add karte hain taake model sirf ek linear rule na seekhe
balke ANN ki non-linear learning capability use ho.

Yeh script standalone bhi chal sakti hai (python models/train_model.py)
aur app.py se bhi automatically call ho sakti hai agar saved model na mile.
"""

import sys
from pathlib import Path

# Taake yeh script directly run karne par bhi project root modules mil jayen
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

from config import (
    FEATURE_NAMES,
    TRAINING_DATA_PATH,
    MODEL_WEIGHTS_PATH,
    SCALER_PATH,
)
from models.ann_model import ResumeFitANN

RANDOM_SEED = 42
NUM_SAMPLES = 4000


def generate_synthetic_dataset(num_samples: int = NUM_SAMPLES) -> pd.DataFrame:
    """Rule-based + noisy synthetic tabular dataset generate karta hai."""
    rng = np.random.default_rng(RANDOM_SEED)

    skill_match = rng.uniform(0, 1, num_samples)
    experience_norm = rng.uniform(0, 1, num_samples)
    education_score = rng.choice([0.2, 0.35, 0.4, 0.6, 0.75, 0.8, 1.0], num_samples)
    text_similarity = rng.uniform(0, 1, num_samples)
    resume_quality = rng.uniform(0, 1, num_samples)

    # Weighted combination se "true fit score" banate hain
    weighted_score = (
        skill_match * 0.35
        + experience_norm * 0.25
        + education_score * 0.15
        + text_similarity * 0.15
        + resume_quality * 0.10
    )

    # Noise add karte hain taake real world jaisi uncertainty aaye
    noise = rng.normal(0, 0.07, num_samples)
    final_score = np.clip(weighted_score + noise, 0, 1)

    # Threshold se binary label banate hain
    labels = (final_score >= 0.55).astype(int)

    df = pd.DataFrame(
        {
            "skill_match_score": skill_match,
            "experience_years_norm": experience_norm,
            "education_score": education_score,
            "text_similarity_score": text_similarity,
            "resume_quality_score": resume_quality,
            "label": labels,
        }
    )
    return df


def train_and_save_model(verbose: bool = True):
    """Poora training pipeline: data generate -> scale -> train -> save."""
    torch.manual_seed(RANDOM_SEED)

    # Step 1: Synthetic dataset generate aur save karo
    df = generate_synthetic_dataset()
    df.to_csv(TRAINING_DATA_PATH, index=False)

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].values.astype(np.float32).reshape(-1, 1)

    # Step 2: Scaling (0-1 range mein, features waise bhi already normalized hain
    # lekin scaler save karna future-proofing ke liye acha practice hai)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_SEED
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    # Step 3: Model, loss aur optimizer setup karo
    model = ResumeFitANN(input_dim=len(FEATURE_NAMES))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 60
    batch_size = 64
    num_batches = int(np.ceil(len(X_train_t) / batch_size))

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        epoch_loss = 0.0

        for i in range(num_batches):
            indices = permutation[i * batch_size : (i + 1) * batch_size]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t)
                val_loss = criterion(val_preds, y_val_t).item()
                val_acc = ((val_preds >= 0.5).float() == y_val_t).float().mean().item()
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_loss / num_batches:.4f} "
                f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

    # Step 4: Model aur scaler disk par save karo
    MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    joblib.dump(scaler, SCALER_PATH)

    if verbose:
        print(f"\nModel saved to: {MODEL_WEIGHTS_PATH}")
        print(f"Scaler saved to: {SCALER_PATH}")

    return model, scaler


if __name__ == "__main__":
    train_and_save_model(verbose=True)
