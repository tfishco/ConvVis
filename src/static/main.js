var convData;
$("#form-num").submit(function(e) {
    $.ajax({
           type: "POST",
           url: "/conv",
           data: $("#form-num").serialize(),
           success: function(data) {
             convData = data;
             var jsondata = JSON.parse(data);
             document.getElementById("prediction").innerHTML= jsondata.convdata.prediction;
             document.getElementById("actual").innerHTML= jsondata.label;
           }
         });
    e.preventDefault();
});
