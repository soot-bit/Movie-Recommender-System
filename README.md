![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2Fm_oepeng)

# Recommender System

This repository contains a recommender system implemented using NumPy, Jax and CUDA. The system uses collaborative filtering techniques to generate personalized recommendations based on user-item interaction data.


## Features

- **Collaborative Filtering**: Implements matrix factorization to learn user and item representations.
- **Alternating Least Squares (ALS)**: Parallisable optimising
- **Datastructrue** A custom data structure usign hash tables and indexing for fast retrieval
- **Bias Handling**: Incorporates user and item biases for improved recommendation accuracy.


## Installation

To set up the recommender system, follow these steps:

1. **Clone the repository:**
2. **Train the model:**
3. **Make predictions:**


   
## File Structure

```
MovieRecommender/
│
├── src/                   
│   ├── cpp/                 # C++/CUDA backend (ongoing .. )
│   │   |
│   │   ├──  # Header file with declarations
│   │   ├──  # CUDA code gpu
│   │   └── Makefile         
│   ├── jax/                 # JAX implementation
│   │   └── model.py 
│   │           
│   ├── bindings/          
│   │   ├── recommender_pybind.cpp 
│   │   └── setup.py        
│   └── common/              # Shared utilities 
│       ├── data_loader.py  
│       └── config.py        
│
├── main.py/                  # Main 
│   ├── train           
│   ├── recommend       
│   ├── benchmark       #  C++/CUDA vs JAX
│   └── config.yaml        
│
├── Data/                # Movie data and model storage
│   ├── 100k.csv         
│   ├── 25M.csv        
│   └── models/            
│
└── README.md             
```

[link for 25 million dataset](http://files.grouplens.org/datasets/movielens/ml-25m.zip)
