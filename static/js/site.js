
$(document).ready(function() {
    $(".form-ajax").on('submit', function (e) {
        var form = $(this)
        e.preventDefault();
        var url = form.attr('action');
        var data = form.serialize();
        $.post(url, data, function (result) {
            if(form.find('.form-results-text').length > 0){
                form.find('.form-results-text').text(result);
                form.find('.form-results-text').show();
            }
            else if(form.find('.form-results-img').length > 0){
                pathArray = result.split(',');
                imgArray = form.find('.form-results-img img');
                for (let i = 0; i < imgArray.length; i++) {
                    var imgurl = pathArray[i] + '?' + new Date().getTime(); // force img refresh
                    $(imgArray[i]).attr("src", imgurl);
                } 
                form.find('.form-results-img').show();
            }
        });
    });
});
