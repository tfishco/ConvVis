def get_layer_coords(feature_size, image_gap_size, no_columns, layer):
    no_rows = len(layer) / no_columns
    y_offset = 0
    x_offset = 0
    dx = x_offset
    dy = y_offset
    x_vals = []
    y_vals = []
    for i in range(len(layer)):
        if i % (no_rows) == 0:
            dx += image_gap_size + feature_size
            dy = y_offset
        else :
            dy += image_gap_size + feature_size
        x_vals.append(dx)
        y_vals.append(dy)
    return x_vals, y_vals

#internal conv layers
#[i] = layer in graph
#[i][0] = no of images in layer
#[i][1] = image height AND width in layer
layers = [32,32,64,64]
layers_size = [28,14,14,7]

def get_coords(image_gap_size, no_columns, layers, layers_size):
    y_offset = 0
    x_offset = 0
    dx = x_offset
    dy = y_offset
    layer_offset = 100
    x_vals = []
    y_vals = []
    for i in range(len(layers)):
        no_rows = layers[i] / no_columns
        for j in range(layers[i]):
            if j % (no_rows) == 0:
                dx += image_gap_size + layers_size[i]
                dy = y_offset
            else :
                dy += image_gap_size + layers_size[i]
            x_vals.append(dx)
            y_vals.append(dy)
        dx += layer_offset
    return x_vals, y_vals

x,y = get_coords(5, 4, layers,layers_size)

for i in range(len(x)):
    print i,x[i],y[i]
