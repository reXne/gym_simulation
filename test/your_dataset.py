# Removing decimals for 'Valor Promedio por línea' and adding monetary nomenclature, also updating labels accordingly

# Create a single figure with horizontal subplots
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# Define a function to add value labels on top of the bars for Valor Promedio por línea with monetary format
def add_value_labels_monetary(axis, bars, data):
    for bar, value in zip(bars, data):
        axis.annotate(f'${value:,.0f}',
                      xy=(bar.get_x() + bar.get_width() / 2, value),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')

# Plotting Valor por línea with monetary value labels and simplified y-axis label
bars0 = ax[0].bar(positions, data['valor_por_linea'].values(), width=bar_width, color=colors)
add_value_labels_monetary(ax[0], bars0, data['valor_por_linea'].values())
ax[0].set_title('Valor por línea')
ax[0].set_ylabel('Valor')
ax[0].set_xticks(positions)
ax[0].set_xticklabels(labels, rotation=45, ha='right')

# Plotting Q líneas por cliente with value labels without changes
bars1 = ax[1].bar(positions, data['q_lineas_por_cliente'].values(), width=bar_width, color=colors)
add_value_labels(ax[1], bars1)
ax[1].set_title('Líneas por cliente')
ax[1].set_ylabel('Líneas por cliente')
ax[1].set_xticks(positions)
ax[1].set_xticklabels(labels, rotation=45, ha='right')

# Making layout tight
plt.tight_layout()

# Save the plot as a single file
file_path_final_adjustments = '/mnt/data/final_adjustments_horizontal_plots.png'
plt.savefig(file_path_final_adjustments, bbox_inches='tight', pad_inches=0.1)

# Close the figure to avoid display in the output
plt.close()

# Return the path to the saved plot file
file_path_final_adjustments
