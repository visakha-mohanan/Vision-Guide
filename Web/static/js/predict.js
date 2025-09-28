// Web/static/js/predict.js

$(document).ready(function() {
    $('#upload-form').on('submit', function(event) {
        event.preventDefault(); // Prevent the default form submission

        var formData = new FormData(this); // Get the form data, including files

        // Clear previous results and show a loading message
        $('#left-eye-result').text('Analyzing...');
        $('#right-eye-result').text('Analyzing...');

        // Send the data to the Flask backend
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            processData: false, // Tell jQuery not to process the data
            contentType: false, // Tell jQuery not to set the content type
            success: function(response) {
                // Update the results on the page
                $('#left-eye-result').text(response['left-eye-prediction']);
                $('#right-eye-result').text(response['right-eye-prediction']);
            },
            error: function(error) {
                console.log(error);
                $('#left-eye-result').text('Error: Could not get a prediction.');
                $('#right-eye-result').text('Error: Could not get a prediction.');
            }
        });
    });
});