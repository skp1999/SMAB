# Import the necessary library
import matplotlib.pyplot as plt

# Define x-axis values
x_values = [1, 2, 3, 4, 5, 6]

# Define the two lists representing different metrics
L1 = [0.416, 0.4846, 0.4200, 0.329, 0.277, 0.260]  # Example data for Accuracy
L2 = [0.877, 0.818, 0.769, 0.730, 0.705, 0.693]  # Example data for Cosine Similarity

# Define variance values for L2 (Cosine Similarity)
variance_L2 = [0.0147, 0.0206, 0.0249, 0.0264, 0.0275, 0.0282]  # Example variance values

# Calculate upper and lower bounds for shaded area
L2_upper = [l2 + var for l2, var in zip(L2, variance_L2)]
L2_lower = [l2 - var for l2, var in zip(L2, variance_L2)]

# Create a plot
plt.figure(figsize=(5, 5))  # Set the size of the figure

# Plot the first metric L1 (Accuracy)
plt.plot(x_values, L1, label='Average ASR', marker='o', linestyle='-', color='blue')

# Plot the second metric L2 (Cosine Similarity) with error bars
plt.errorbar(x_values, L2, yerr=variance_L2, label='Average Cosine Similarity', 
             fmt='x', linestyle='--', color='red', ecolor='gray', capsize=5)

# Add shaded area representing the variance for L2
plt.fill_between(x_values, L2_lower, L2_upper, color='red', alpha=0.2)

# Adding titles and labels
plt.title('Average ASR vs Average  Cosine Similarity')
plt.xlabel('Number of Words replaced.')
plt.ylabel('ASR and Cosine Similarity')

# Add a legend to differentiate the metrics
plt.legend()

# Display the grid
plt.grid(True)

# Save the plot as an image file
plt.savefig('metrics_comparison_with_variance.png', bbox_inches='tight')  # Save as 'metrics_comparison_with_variance.png' with high resolution
