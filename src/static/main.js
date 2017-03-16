var p = document.getElementById("net-struct");
p.value = JSON.stringify([
  [1, 32, 32, 64, 64, 1, 1, 10],
  ['input_1', 'conv_1', 'pool_1',
    'conv_2', 'pool_2', 'fc_1', 'fc_2', 'decision_1'
  ]
]);

var convData;

$("#form-num").submit(function(e) {
  $.ajax({
    type: "POST",
    url: "/conv",
    data: $("#form-num").serialize(),
    success: function(data) {
      var jsondata = JSON.parse(data);
      document.getElementById("prediction").innerHTML = jsondata.convdata
        .prediction;
      document.getElementById("actual").innerHTML = jsondata.label;
      gen_graph(jsondata);
    }
  });
  e.preventDefault();
});
