var p = document.getElementById("net-struct");
var struct = [1,32,32,64,64,1,10];
p.value = JSON.stringify(struct);

var convData;

$("#form-num").submit(function(e) {
    $.ajax({
           type: "POST",
           url: "/conv",
           data: $("#form-num").serialize(),
           success: function(data) {
             var jsondata = JSON.parse(data);
             document.getElementById("prediction").innerHTML= jsondata.convdata.prediction;
             document.getElementById("actual").innerHTML= jsondata.label;
             gen_graph(jsondata);
           }
         });
    e.preventDefault();
});
