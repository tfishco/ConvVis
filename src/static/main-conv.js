var p = document.getElementById("net-struct");
p.value = JSON.stringify([
  [1, 32, 32, 64, 64, 1, 1, 10],
  ['input_0', 'conv_0', 'pool_0',
    'conv_1', 'pool_1', 'fc_0', 'fc_1', 'decision_0'
  ]
]);

function toggleElement(id) {
  var e = document.getElementById(id);
  e.style.display = (e.style.display == 'block') ? 'none' : 'block';
}

$("#form-num").submit(function(e) {
  $.ajax({
    type: 'POST',
    url: '/conv',
    data: $("#form-num").serialize(),
    success: function(data) {
      var jsondata = JSON.parse(data);
      gen_graph(jsondata);

      var dn = document.getElementById("dataset-name");
      var i = document.getElementById("iterations");
      dn.value = jsondata.dataset;
      i.value = jsondata.training_iter;

      $("#form-data").submit(function(e) {
        toggleElement('overlay');
        $.ajax({
          type: 'POST',
          url: '/dataset_data',
          data: $("#form-data").serialize(),
          success: function(data) {
            json_data = JSON.parse(data);
            toggleElement('dataset-info');
            e = document.getElementById('dataset-accuracy');
            e.innerHTML = "Test accuracy: " + json_data.test_accuracy;
            b = document.getElementById('close-overlay');
            b.onclick = function() {
              toggleElement('overlay');
              toggleElement('dataset-info');
            };
          }
        });
        e.preventDefault();
      });
    }
  });
  e.preventDefault();
});
