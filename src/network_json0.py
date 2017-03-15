import json
import numpy as np

def get_coords(vert_size_gap, horiz_size_gap, no_columns, layers, layers_size):
    y_offset = 50
    x_offset = 250
    dx = x_offset
    dy = y_offset
    layer_offset = 50
    x_vals = []
    y_vals = []
    for i in range(len(layers)):
        no_rows = layers[i] / no_columns
        for j in range(layers[i]):
            if j % (no_rows) == 0:
                dx += horiz_size_gap + layers_size[i]
                dy = y_offset
            else :
                dy += vert_size_gap + layers_size[i]
            x_vals.append(dx)
            y_vals.append(dy)
        dx += layer_offset
    return x_vals, y_vals

# struct = [1, 32, 32, 64, 64, 1, 1, 10]
# node_type = ['conv_1', 'pool_1','conv_2', 'pool_2', 'fc_1', 'fc_2', 'decision']

def get_json(struct, node_type, value, seperate_conv):
    main = {}
    nodes = []
    links = []
    # Nodes
    pixel_count = 60
    value_count = 0
    count = 0
    x_vals, y_vals = get_coords(25, 15, 4, [32,32,64,64], [28,14,14,7])
    for i in range(len(struct)):
        for j in range(struct[i]):
            node = {}
            node['name'] = str(i) + '_' + str(j)
            if i == 0 or i == len(struct) - 2:
                node['y'] = 330
                node['fixed'] = True
                if i == 0:
                    node['x'] = 210
                else:
                    node['x'] = 1100
            elif i == len(struct) - 1:
                pixel_count += 50
                node['y'] = pixel_count
                node['x'] = 1200
                node['value'] = value[value_count]
                node['fixed'] = True
                value_count += 1
            elif i==len(struct) - 3:
                node['y'] = 330
                node['x'] = 1000
                node['fixed'] = True
            else:
                node['x'] = x_vals[count]
                node['y'] = y_vals[count]
                node['fixed'] = True
                count += 1
            nodes.append(node)
# struct = [1, 32, 32, 64, 64, 1, 1, 10]
# node_type = ['input_0', 'conv_1', 'pool_1','conv_2', 'pool_2', 'fc_1', 'fc_2', 'decision_0']
# seperate_conv =
# separate_conv1
# separate_conv2
# indexes_conv1
# indexes_conv2

    # Links
    for i in range(len(struct)):
        if i < len(struct) - 1 and node_type[i + 1] == 'conv_1':
            brightnesses = seperate_conv['separate_conv' + str(node_type[i + 1].split("_")[1])][0]
            indexes = seperate_conv['indexes_conv1' + + str(node_type[i + 1].split("_")[1])]
            for j in range(len(brightnesses)):
                link = {}
                print(brightnesses[j])
                link['source'] = 0 # 1 - 32
                link['target'] = indexes[j][2] # 33 - 64
                links.append(link)
        if i < len(struct) - 1 and node_type[i + 1] == 'conv_1':
            print(node_type[i + 1])
            brightnesses = seperate_conv['separate_conv' + str(node_type[i + 1].split("_")[1])][0]
            np.array(brightnesses)
            for j in range(len(brightnesses[i])):
                link = {}
                link['source'] = 0
                link['target'] = brightnesses[j][2] # 1 - 32
                links.append(link)

    main['nodes'] = nodes
    main['links'] = links

    return json.dumps(main) , struct
