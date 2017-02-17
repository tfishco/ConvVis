import json
import numpy as np
#struct = [1,32,32,64,64,3136,1024,10] fully connected
#struct = [1,32,32,64,64,1,1,10]

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

def get_json(struct, value):
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
                if i == 0:
                    node['x'] = 210
                else:
                    node['x'] = 1100
                node['fixed'] = True
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

    # Links
    for i in range(len(struct)):
        if i == 0:
            for j in range(1, struct[i + 1] + 1):
                link = {}
                link['source'] = 0
                link['target'] = j # 1 - 32
                links.append(link)
        if i == 1:
            for j in range(1, struct[i] + 1):
                link = {}
                link['source'] = j # 1 - 32
                link['target'] = j + struct[i] # 33 - 64
                links.append(link)
        if i == 2:
            for j in range(1, struct[i] + 1):
                link1 = {}
                link2 = {}
                link1['source'] = j + struct[i]
                link2['source'] = j + struct[i]
                link1['target'] = j + struct[i + 1] + j
                link2['target'] = j + struct[i + 1] + j - 1
                links.append(link1)
                links.append(link2)
        if i == 3:
            for j in range(1,struct[i] + 1):
                link = {}
                link['source'] = j + struct[i]
                link['target'] = j + struct[i] + struct[i + 1]
                links.append(link)
        if i == 4:
            for j in range(1,struct[i] + 1):
                link = {}
                link['source'] = j + struct[i - 1] + struct[i]
                link['target'] = 193
                links.append(link)
        if i == 5:
            link = {}
            link['source'] = 193
            link['target'] = 194
            links.append(link);
        if i == 6:
            for j in range(1,struct[i + 1] + 1):
                link = {}
                link['source'] = 194
                link['target'] = 194 + j
                links.append(link)

    main['nodes'] = nodes
    main['links'] = links

    return json.dumps(main) , struct
