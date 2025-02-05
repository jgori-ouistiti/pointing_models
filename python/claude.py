import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, t
from scipy.stats import multivariate_normal

# Set random seed for reproducibility
np.random.seed(42)

def generate_gaussian_copula_samples(n_samples, rho):
    # Generate correlated normal variables
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    Z = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Transform to uniform using the normal CDF
    U = norm.cdf(Z)
    return U, Z

def plot_copula_examples(n_samples=2000, rho=0.7):
    U, Z = generate_gaussian_copula_samples(n_samples, rho)
    
    # Create different marginal transformations
    X1_normal = Z[:, 0]
    X2_normal = Z[:, 1]
    
    # Gamma margins
    X1_gamma = gamma.ppf(U[:, 0], a=2)
    X2_gamma = gamma.ppf(U[:, 1], a=2)
    
    # Student's t margins (df=3)
    X1_t = t.ppf(U[:, 0], df=3)
    X2_t = t.ppf(U[:, 1], df=3)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot 1: Scatter in normal space
    axes[0,0].scatter(X1_normal, X2_normal, alpha=0.5, s=2)
    axes[0,0].set_title('Normal Margins (Original Copula Space)')
    axes[0,0].set_xlabel('X1')
    axes[0,0].set_ylabel('X2')
    
    # Plot 2: Contour in normal space
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal([0, 0], [[1, rho], [rho, 1]])
    Z_contour = rv.pdf(pos)
    
    axes[0,1].contour(X, Y, Z_contour, levels=20)
    axes[0,1].set_title('Density Contours in Normal Space')
    axes[0,1].set_xlabel('X1')
    axes[0,1].set_ylabel('X2')
    
    # Plot 3: Gamma margins
    axes[1,0].scatter(X1_gamma, X2_gamma, alpha=0.5, s=2)
    axes[1,0].set_title('Gamma Margins')
    axes[1,0].set_xlabel('X1')
    axes[1,0].set_ylabel('X2')
    
    # Plot 4: Student's t margins
    axes[1,1].scatter(X1_t, X2_t, alpha=0.5, s=2)
    axes[1,1].set_title('Student t Margins (df=3)')
    axes[1,1].set_xlabel('X1')
    axes[1,1].set_ylabel('X2')
    
    plt.tight_layout()
    return fig

# Generate the plots
plot_copula_examples()
