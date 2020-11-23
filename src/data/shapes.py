import io
import matplotlib.pyplot as plt

def make_shape(shape = 'square', color = 'red', size = 50):
	# Define figure
	fig = plt.figure(figsize = (10,10))
	ax = fig.gca()

	# Set background and plotting area to black, remove axis
	plt.axis('off')
 

	# Draw shape depending on type
	if shape = 'square':
		# Draw square
		# Calculate width and height
		square_size = size/100
		# Calculate starting coordinates as rectangles are drawn from the bottom left corner
		starting_pos = (1 - square_size) / 2
		circle = plt.Rectangle(xy = (starting_pos, starting_pos), width = square_size, height = square_size, color = color)
		ax.add_artist(circle)		
	if shape = 'circle':
		# Draw circle
		circle = plt.Circle(xy = (0.5, 0.5), radius = 0.5*size/100, color = color)
		ax.add_artist(circle)

	# Save figure to buffer
	buffer = io.BytesIO()
	plt.savefig(buffer, format='png', facecolor = 'black')

	# Close figure
	plt.close()

	# Return pointer to buffered image
	buffer.seek(0)
	return buffer.getvalue()

def make_square(color = 'red', size = 50):
	return make_shape(shape = 'square', color, size)

def make_circle(color = 'red', size = 50):
	return make_shape(shape = 'circle', color, size)