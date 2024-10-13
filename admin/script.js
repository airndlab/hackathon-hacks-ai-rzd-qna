document.getElementById('rebuild-btn').addEventListener('click', function() {
    // Здесь укажите URL для отправки данных
    const url = 'http://176.123.163.193:8080/api/indexing';

    axios.post(url)
    .then(function(response) {
        console.log('Индекс перестроен:', response.data);
        alert('Индекс успешно перестроен!');
    })
    .catch(function(error) {
        console.error('Ошибка:', error);
        alert('Произошла ошибка при перестройке индекса.');
    });
});
