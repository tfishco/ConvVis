import json
import numpy as np
#struct = [1,32,32,64,64,3136,1024,10] fully connected
#struct = [1,32,32,64,64,1]

def get_json(struct, value):
    main = {}
    nodes = []
    links = []
    # Nodes
    pixel_count = 60
    value_count = 0
    for i in range(len(struct)):
        for j in range(struct[i]):
            node = {}
            node['name'] = str(i) + '_' + str(j)
            if i == 0 or i == len(struct) - 2:
                node['fixed'] = True
                node['y'] = 330
                if i == 0:
                    node['x'] = 210
                else:
                    node['x'] = 1000
            elif i == len(struct) - 1:
                pixel_count += 50
                node['fixed'] = True
                node['y'] = pixel_count
                node['x'] = 1150
                node['value'] = value[value_count]
                value_count += 1
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
            for j in range(1,struct[i + 1] + 1):
                link = {}
                link['source'] = 193
                link['target'] = 193 + j
                links.append(link)

    main['nodes'] = nodes
    main['links'] = links

    return json.dumps(main) , struct
