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

$("#form-num").validate({
  rules: {
    val: {
      required: true,
      range: [0, 9999]
    }
  },
  messages: {
    val: "Enter a number between 0 and 9999"
  },
  submitHandler: function(form) {
    console.log(form);
    $.ajax({
      type: 'POST',
      url: '/conv',
      data: $(form).serialize(),
      success: function(data) {
        var jsondata = JSON.parse(data);
        gen_graph(jsondata);
      }
    });
  }
});
