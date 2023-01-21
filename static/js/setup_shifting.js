
$(document).ready(function() { // executes the below code when the page finishes loading

    $("#saturation").on('change', function (e) {
        $("#saturation-label").text(this.value);
        $(this.form).trigger('submit');
    });

    $("#value").on('change', function (e) {
        $("#value-label").text(this.value);
        $(this.form).trigger('submit');
    });
    
    $("#openkernel").on('change', function (e) {
        $("#openkernel-label").text(this.value);
        $(this.form).trigger('submit');
    });

    $("#houghthres").on('change', function (e) {
        $("#houghthres-label").text(this.value);
        $(this.form).trigger('submit');
    });

    $("#dilationkernel").on('change', function (e) {
        $("#dilationkernel-label").text(this.value);
        $(this.form).trigger('submit');
    });
  
});


