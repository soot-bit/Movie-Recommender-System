from src.trainer import Trainer
from src.py.vanilla.als import ALSModel
from src.py.jax.model import ALSJax


def main():
    trainer = Trainer()

    epochs = 20
    model = trainer.fit(ALSModel, epochs)

    user_idx = 1  
    movie_idx = 10 
    predicted_rating = model.predict(user_idx, movie_idx)

    print(f"Predicted rating for User {user_idx} on Movie {movie_idx}: {predicted_rating:.2f}")

if __name__ == '__main__':
    main()