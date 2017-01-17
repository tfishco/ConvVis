var data = decision;
console.log(data)

var x = d3.scale.linear()
    .domain([0, d3.max(data)])
    .range([0, 200]);

d3.select(".chart")
  .selectAll("div")
    .data(data)
  .enter().append("div")
    .style("width", function(d) { return x(d) + 50 + "px"; })
    .text(function(d) { return d  + "%"; });
