function gen_graph(data) {

  var width = 1275,
    height = 660;

  var color = d3.scale.category20();

  var force = d3.layout.force()
    .charge(function(d) {
      var charge = -100;
      if (d.index === 0) charge = 10 * charge;
      return charge;
    })
    .linkDistance(30)
    .size([width, height]);

  var svg = d3.select(".canvas").append("svg")
    .attr("width", width)
    .attr("height", height);

  var tip = d3.tip()
    .attr('class', 'd3-tip')
    .offset([-10, 0])
    .attr("class", "tooltip")
    .html(function(d) {
      return d.name + "</span>";
    })
  svg.call(tip);

  graph = JSON.parse(data.struct)

  force.nodes(graph.nodes)
    .links(graph.links)
    .start();

  var link = svg.selectAll(".link")
    .data(graph.links)
    .enter().append("line")
    .attr("class", "link")
    .style("stroke-width", function(d) {
      return Math.sqrt(d.value);
    });

  var node = svg.selectAll(".node")
    .data(graph.nodes)
    .enter().append("g")
    .attr("class", function(d) {
      var image_ref = d.name.split("_");
      if (image_ref[0] < 5) {
        return "node-image";
      } else {
        return "node-plain";
      }
    })
    .call(force.drag)
    .on('dblclick', connectedNodes);

  var image = d3.selectAll(".node-image")
    .append("image")
    .attr("x", -8)
    .attr("y", -8)
    .attr("id", function(d) {
      return "image_" + d.name;
    })
    .call(populateNodes);

  var plainNode = d3.selectAll(".node-plain")
    .append("circle")
    .attr("class", "circle-node")
    .attr("r", function(d) {
      var image_ref = d.name.split("_");
      if (parseInt(image_ref[0]) < 7) {
        return 2;
      } else {
        var val = data.convdata.log_certainty[parseInt(image_ref[1])] * 25
        if (val < 2) {
          return 2;
        } else {
          return val;
        }
      }
    })
    .attr("fill", function(d) {
      var image_ref = d.name.split("_");
      console.log(d.index, data.label);
      if (image_ref[1] == data.convdata.prediction) {
        return 'orangered';
      } else {
        return 'orange';
      }
    });

  force.on("tick", function() {
    link.attr("x1", function(d) {
        return d.source.x;
      })
      .attr("y1", function(d) {
        return d.source.y;
      })
      .attr("x2", function(d) {
        return d.target.x;
      })
      .attr("y2", function(d) {
        return d.target.y;
      });

    node.attr("transform", function(d) {
      return "translate(" + d.x + "," + d.y + ")";
    });
  });

  var toggle = 0;

  var linkedByIndex = {};
  for (i = 0; i < graph.nodes.length; i++) {
    linkedByIndex[i + "," + i] = 1;
  };
  graph.links.forEach(function(d) {
    linkedByIndex[d.source.index + "," + d.target.index] = 1;
  });

  function populateNodes() {
    var nodes = d3.selectAll("image")[0];
    for (i = 0; i < nodes.length; i++) {
      var image_ref = nodes[i].__data__.name.split("_");
      if (parseInt(image_ref[0]) < 5) {
        var raw = data.convdata.features[(parseInt(image_ref[0]) + 1)][
          image_ref[1]
        ]["feature_" + parseInt(image_ref[1])];
        var buffer = new Uint8ClampedArray(raw);

        var width = Math.sqrt(raw.length / 4),
          height = Math.sqrt(raw.length / 4);

        var canvas = document.createElement("canvas");
        //console.log(canvas, canvas.nodeName);
        var ctx = canvas.getContext('2d');

        canvas.width = width;
        canvas.height = height;

        var idata = ctx.createImageData(width, height);
        idata.data.set(buffer);
        ctx.putImageData(idata, 0, 0);

        //set image created by canvas to image element.
        var image = document.getElementById("image_" + nodes[i].__data__.name);
        image.width.baseVal.value = width;
        image.height.baseVal.value = height;
        image.href.baseVal = canvas.toDataURL();
      }
    }
  }

  function neighboring(a, b) {
    return linkedByIndex[a.index + "," + b.index];
  }

  function connectedNodes() {
    if (toggle == 0) {
      d3.select(this).fixed = true;
      d = d3.select(this).node().__data__;
      node.style("opacity", function(o) {
        return neighboring(d, o) | neighboring(o, d) ? 1 : 0.1;
      });

      link.style("opacity", function(o) {
        return d.index == o.source.index | d.index == o.target.index ? 1 :
          0.1;
      });

      toggle = 1;
    } else {
      node.style("opacity", 1);
      link.style("opacity", 1);
      toggle = 0;
    }
  }

  force.start();

}
