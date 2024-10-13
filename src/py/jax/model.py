import jax.numpy as jnp
from jax import jit, vmap
from jax import random

class ALSJax:
    """ ALS in JAX """
    def __init__(self, user_data, item_data, latent_d=5, lamda=0.001, tau=0.01, gamma=0.001, seed=0):
        self.user_data = user_data
        self.item_data = item_data
        self.lamda = lamda
        self.tau = tau
        self.gamma = gamma
        self.latent_d = latent_d
        
        
        key = random.PRNGKey(seed)
        self.user_matrix = random.normal(key, (len(user_data), latent_d)) * (1 / jnp.sqrt(latent_d))
        self.item_matrix = random.normal(key, (len(item_data), latent_d)) * (1 / jnp.sqrt(latent_d))
        self.user_bias = jnp.zeros((len(user_data),))
        self.item_bias = jnp.zeros((len(item_data),))

    def _update_bias_and_matrix(self, matrix, biases, data, lamda, tau, latent_d, is_user):
        """ 
        Update bias and matrix for users or items.
        If is_user is True, it updates user_matrix and user_bias; otherwise, it updates item_matrix and item_bias.
        """
        def update_single(idx):
            if not data[idx]:
                return matrix[idx, :], biases.at[idx].get()  

            r = jnp.array([r for n, r in data[idx]])
            n = jnp.array([n for n, r in data[idx]])
            inner_product = jnp.dot(matrix[idx, :], matrix[n, :].T) if is_user else jnp.dot(matrix[n, :], matrix[idx, :].T)
            bias = lamda * jnp.sum(r - (inner_product + biases[n]))

            # Update bias
            biases = biases.at[idx].set(bias / (lamda * len(r) + self.gamma))

            # Update matrix
            if is_user:
                b_matrix = lamda * jnp.sum(self.item_matrix[n, :] * (r - (biases[idx] + biases[n])), axis=0)
                inv_matrix = lamda * jnp.sum(jnp.outer(self.item_matrix[n, :], self.item_matrix[n, :]), axis=0) + tau * jnp.eye(latent_d)
            else:
                b_matrix = lamda * jnp.sum(self.user_matrix[n, :] * (r - (biases[n] + biases[idx])), axis=0)
                inv_matrix = lamda * jnp.sum(jnp.outer(self.user_matrix[n, :], self.user_matrix[n, :]), axis=0) + tau * jnp.eye(latent_d)

            new_matrix = jnp.linalg.solve(inv_matrix, b_matrix)
            return new_matrix, biases

        return vmap(update_single)(jnp.arange(len(data)))

    @jit
    def train(self, iterations):
        for _ in range(iterations):
            self.user_matrix, self.user_bias = self._update_bias_and_matrix(
                self.user_matrix, self.user_bias, self.user_data, self.lamda, self.tau, self.latent_d, is_user=True
            )
            
            self.item_matrix, self.item_bias = self._update_bias_and_matrix(
                self.item_matrix, self.item_bias, self.item_data, self.lamda, self.tau, self.latent_d, is_user=False
            )

    @jit
    def predict(self, m, n):
        return jnp.dot(self.user_matrix[m, :], self.item_matrix[n, :]) + self.user_bias[m] + self.item_bias[n]
