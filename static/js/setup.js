
$(document).ready(function() {
    
    $("#vs").on('change', function (e) {
        $("#vs-label").text(this.value);
        $(this.form).trigger('submit');
    });

    $("#blur_T").on('change', function (e) {
        $("#blur_T-label").text(this.value);
        $(this.form).trigger('submit');
    });

});


