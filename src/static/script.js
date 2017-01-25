function gen_graph(data) {

  var width = 1075,
      height = 660;

  var color = d3.scale.category20();

  var force = d3.layout.force()
      .charge(-150)
      .linkDistance(30)
      .size([width, height]);

  var svg = d3.select(".canvas").append("svg")
      .attr("width", width)
      .attr("height", height);

  var tip = d3.tip()
      .attr('class', 'd3-tip')
      .offset([-10, 0])
      .attr("class", "tooltip")
      .html(function (d) {
      return  d.name + "</span>";
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
      .style("stroke-width", function (d) {
      return Math.sqrt(d.value);
  });

  var node = svg.selectAll(".node")
      .data(graph.nodes)
      .enter().append("circle")
      .attr("class", "node")
      .attr("r", 8)
      .style("fill", "#FCFCFC")
      .call(force.drag)
      .on('click', onNodeClick);
      //.on('mouseover', tip.show) //Added
      //.on('mouseout', tip.hide) //Added
      //.on('dblclick', connectedNodes);
      //.on('click', updateImage)


  force.on("tick", function () {
      link.attr("x1", function (d) {
          return d.source.x;
      })
          .attr("y1", function (d) {
          return d.source.y;
      })
          .attr("x2", function (d) {
          return d.target.x;
      })
          .attr("y2", function (d) {
          return d.target.y;
      });

      node.attr("cx", function (d) {
          return d.x;
      })
          .attr("cy", function (d) {
          return d.y;
      });
  });

  var toggle = 0;

  var linkedByIndex = {};
  for (i = 0; i < graph.nodes.length; i++) {
      linkedByIndex[i + "," + i] = 1;
  };
  graph.links.forEach(function (d) {
      linkedByIndex[d.source.index + "," + d.target.index] = 1;
  });

  function neighboring(a, b) {
      return linkedByIndex[a.index + "," + b.index];
  }

  function connectedNodes() {
      if (toggle == 0) {
          d3.select(this).fixed = true;
          d = d3.select(this).node().__data__;
          node.style("opacity", function (o) {
              return neighboring(d, o) | neighboring(o, d) ? 1 : 0.1;
          });

          link.style("opacity", function (o) {
              return d.index==o.source.index | d.index==o.target.index ? 1 : 0.1;
          });

          toggle = 1;
      } else {
          node.style("opacity", 1);
          link.style("opacity", 1);
          toggle = 0;
      }
  }

  var count = 0
  for (i = 1; i <= 6; i++) {
    for (j = 0; j < data.no_nodes[i - 1]; j++){
      count += 1
      var brightness = data.convdata.features[i]["" + data.no_nodes[i - 1] + ""][j];
      //d3.select(d3.selectAll(".node"))[0][0][0][count].style.brightness = brightness;
      d3.select(d3.selectAll(".node"))[0][0][0][count].style.opacity = (brightness/50);
    }
  }
  console.log(count);

  function onNodeClick() {

    var image_ref = d3.select(this)[0][0].__data__.name.split("_");

    console.log(parseInt(image_ref[0]) + 1, image_ref[1]);

    var raw = data.convdata.features[(parseInt(image_ref[0]) + 1)][image_ref[1]]["feature_" + parseInt(image_ref[1])];

    console.log(raw);

    var buffer = new Uint8ClampedArray(raw);

    var width = Math.sqrt(raw.length / 4),
      height = Math.sqrt(raw.length / 4);

      console.log(width, height);

    var canvas = document.getElementById('image'),
      ctx = canvas.getContext('2d');

    canvas.width = width;
    canvas.height = height;
    var idata = ctx.createImageData(width, height);
    idata.data.set(buffer);
    ctx.putImageData(idata, 0, 0);

    document.getElementById("image-height").innerHTML = height;
    document.getElementById("image-width").innerHTML = width;


  }

}
